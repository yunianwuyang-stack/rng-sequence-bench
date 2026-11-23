"""Participant-level forecast summaries and calibration diagnostics."""

import numpy as np
from scipy.optimize import isotonic_regression, minimize
from scipy.special import logsumexp

SCORE_NAMES = ("log_loss", "brier", "top1_accuracy")


def _validate_mask(mask, shape):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.shape != shape or m.dtype != bool:
        raise ValueError("Mask must be a boolean array matching participant-by-trial shape")
    if not m.any(axis=1).all():
        raise ValueError("Each participant must have at least one unmasked trial")
    return m


def scores(prob, target, mask=None):
    """Return per-participant (log_loss, brier, top1_accuracy) means.

    When ``mask`` is supplied (boolean array matching ``target.shape``), only
    unmasked trials contribute to each participant's mean.
    """
    prob = np.asarray(prob, dtype=float)
    target = np.asarray(target)
    m = _validate_mask(mask, target.shape)
    safe_target = np.where(m, target, 0) if m is not None else target
    actual = np.take_along_axis(prob, safe_target[..., None], axis=-1)[..., 0]
    maximum = prob.max(axis=-1)
    ties = np.isclose(prob, maximum[..., None], rtol=0, atol=1e-12).sum(axis=-1)
    hit = np.isclose(maximum, actual, rtol=0, atol=1e-12) / ties
    per_trial = np.stack(
        (
            -np.log(np.maximum(actual, 1e-300)),
            (prob**2).sum(axis=-1) - 2 * actual + 1,
            hit,
        ),
        axis=-1,
    )
    if m is None:
        return per_trial.mean(axis=1)
    weights = m[..., None].astype(float)
    return (per_trial * weights).sum(axis=1) / weights.sum(axis=1)


def interval(values, rng, reps=3000):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        mean = arr[0].tolist()
        return {"mean": mean, "ci95": [mean, mean]}
    draws = np.concatenate(
        [
            arr[
                rng.integers(len(arr), size=(min(100, reps - start), len(arr)))
            ].mean(axis=1)
            for start in range(0, reps, 100)
        ]
    )
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "ci95": np.quantile(draws, [0.025, 0.975], axis=0).tolist(),
    }


def reliability(probabilities, outcomes, *, n_bins=10, strategy="uniform"):
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if strategy == "uniform":
        group = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    elif strategy == "quantile":
        edges = np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1))
        group = np.searchsorted(edges[1:-1], probabilities, side="right")
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    n = np.bincount(group, minlength=n_bins)
    predicted = np.bincount(group, weights=probabilities, minlength=n_bins)
    actual = np.bincount(group, weights=outcomes, minlength=n_bins)
    rows = [
        {
            "bin": i,
            "n": int(n[i]),
            "predicted": float(predicted[i] / n[i]) if n[i] else None,
            "observed": float(actual[i] / n[i]) if n[i] else None,
        }
        for i in range(n_bins)
    ]
    return rows, float(np.abs(predicted - actual).sum() / len(probabilities))


def corp_reliability(probabilities, outcomes):
    """Isotonic (CORP) reliability summary following Dimitriadis et al. (2021)."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    # Pool ties in predicted probabilities before PAV isotonic regression.
    uniq_p, idx, counts = np.unique(p_sorted, return_index=True, return_counts=True)
    sums = np.add.reduceat(y_sorted, idx)
    means = sums / counts
    iso = isotonic_regression(means, weights=counts.astype(float))
    fitted = np.repeat(iso.x, counts)
    miscalibration = float(np.mean(np.abs(p_sorted - fitted)))
    brier_miscal = float(np.mean((p_sorted - fitted) ** 2))
    return {
        "corp_ece": miscalibration,
        "corp_brier_miscalibration": brier_miscal,
        "unique_levels": int(len(uniq_p)),
    }


def calibration(prob, target, mask=None):
    n_classes = prob.shape[-1]
    m = _validate_mask(mask, target.shape)
    if m is None:
        prob_flat = prob.reshape(-1, n_classes)
        target_flat = target.ravel()
    else:
        prob_flat = prob[m]
        target_flat = target[m]
    n = len(target_flat)
    events = np.eye(n_classes)[target_flat]
    one = scores(prob_flat[None], target_flat[None])[0]
    category_error = []
    category_bins = []
    rel = res = unc = coarsened = 0.0
    for c in range(n_classes):
        rows, error = reliability(prob_flat[:, c], events[:, c])
        base = events[:, c].mean()
        unc += base * (1 - base)
        for row in rows:
            if row["n"]:
                size = row["n"] / n
                p, q = row["predicted"], row["observed"]
                rel += size * (p - q) ** 2
                res += size * (q - base) ** 2
                coarsened += size * (p * p - 2 * p * q + q)
        category_bins.append(rows)
        category_error.append(error)
    if abs(coarsened - (rel - res + unc)) > 1e-10:
        raise RuntimeError("Binned Brier decomposition does not reconstruct the score")

    log_p = np.log(np.maximum(prob_flat, 1e-300))

    def objective(parameters):
        slope, offsets = parameters[0], parameters[1:]
        eta = slope * log_p + offsets
        normalizer = logsumexp(eta, axis=-1)
        delta = np.exp(eta - normalizer[:, None]) - events
        loss = (normalizer - eta[np.arange(n), target_flat]).mean()
        gradient = np.r_[(delta * log_p).sum() / n, delta.mean(axis=0)]
        return loss, gradient

    slope = offsets = None
    relative_logits = log_p - log_p.mean(axis=1, keepdims=True)
    if events.sum(axis=0).min() == 0:
        fit_status = "missing_observed_category"
    elif np.max(np.ptp(relative_logits, axis=0)) <= 1e-12:
        fit_status = "constant_forecasts"
    else:
        fit = minimize(
            objective,
            np.r_[1.0, np.zeros(n_classes)],
            jac=True,
            method="L-BFGS-B",
            options={"gtol": 1e-8, "ftol": 1e-13},
        )
        if not np.isfinite(fit.x).all() or np.max(np.abs(fit.jac)) > 1e-6:
            fit_status = "diagnostic_fit_failed"
        else:
            slope = float(fit.x[0])
            offsets = (fit.x[1:] - fit.x[1:].mean()).tolist()
            fit_status = "estimated"
    picked = prob_flat[np.arange(n), target_flat]
    confidence = prob_flat.max(axis=1)
    ties = np.isclose(prob_flat, confidence[:, None], atol=1e-12, rtol=0).sum(axis=1)
    hit = np.isclose(picked, confidence, atol=1e-12, rtol=0) / ties
    top_bins, top_ece = reliability(confidence, hit)
    corp = corp_reliability(confidence, hit)
    cal_in_large = (prob_flat.mean(axis=0) - events.mean(axis=0)).tolist()
    return {
        "score_names": list(SCORE_NAMES),
        "scores": one.tolist(),
        "named_scores": {
            name: float(val) for name, val in zip(SCORE_NAMES, one, strict=True)
        },
        "top_ece": top_ece,
        "top_nonempty_bins": int(sum(1 for row in top_bins if row["n"] > 0)),
        "top_bins": top_bins,
        "top_corp": corp,
        "mean_class_ece": float(np.mean(category_error)),
        "class_ece": category_error,
        "class_bins": category_bins,
        "calibration_in_the_large": cal_in_large,
        "shared_logit_slope": slope,
        "centered_class_offsets": offsets,
        "calibration_fit_status": fit_status,
        "brier": {
            "raw": float(one[1]),
            "reliability": rel,
            "resolution": res,
            "uncertainty": unc,
            "within_bin_correction": float(one[1] - coarsened),
        },
    }


def forecast_diagnostics(probabilities, targets, *, mask=None):
    """Score fixed forecasts for participant-by-trial trajectories.

    Probabilities have shape ``(participants, trials, categories)`` with strictly
    positive, normalized rows on all unmasked trials. Targets are zero-based
    integers with shape ``(participants, trials)``. Pass an optional boolean
    ``mask`` of shape ``(participants, trials)`` to handle unequal sequence
    lengths or missing responses.
    """
    prob = np.asarray(probabilities, dtype=float)
    target = np.asarray(targets)
    if prob.ndim != 3 or min(prob.shape) == 0 or prob.shape[-1] < 2:
        raise ValueError(
            "Expected nonempty participant-by-trial-by-category probabilities"
        )
    if target.shape != prob.shape[:2] or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("Targets must be integer participant-by-trial indices")
    m = _validate_mask(mask, target.shape)
    valid_target = target if m is None else target[m]
    valid_prob = prob if m is None else prob[m]
    if valid_target.min() < 0 or valid_target.max() >= prob.shape[-1]:
        raise ValueError("Targets must be zero-based indices within the alphabet")
    if (
        not np.isfinite(valid_prob).all()
        or (valid_prob <= 0).any()
        or not np.allclose(valid_prob.sum(axis=-1), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("Forecasts must be finite, positive and sum to one")
    return calibration(prob, target, mask=m)


def ece_binning_sensitivity(probabilities, targets, *, bin_counts=(5, 10, 15), mask=None):
    """Compare top-choice and classwise ECE across equal-width and quantile bins plus CORP."""
    diag = forecast_diagnostics(probabilities, targets, mask=mask)
    prob = np.asarray(probabilities, dtype=float)
    target = np.asarray(targets)
    m = _validate_mask(mask, target.shape)
    prob_flat = prob.reshape(-1, prob.shape[-1]) if m is None else prob[m]
    target_flat = target.ravel() if m is None else target[m]
    n_classes = prob.shape[-1]
    events = np.eye(n_classes)[target_flat]
    picked = prob_flat[np.arange(len(target_flat)), target_flat]
    confidence = prob_flat.max(axis=1)
    ties = np.isclose(prob_flat, confidence[:, None], atol=1e-12, rtol=0).sum(axis=1)
    hit = np.isclose(picked, confidence, atol=1e-12, rtol=0) / ties

    out = {"uniform": {}, "quantile": {}, "corp": diag["top_corp"]}
    for strategy in ("uniform", "quantile"):
        for b in bin_counts:
            top_rows, top_e = reliability(confidence, hit, n_bins=b, strategy=strategy)
            class_e = [
                reliability(prob_flat[:, c], events[:, c], n_bins=b, strategy=strategy)[1]
                for c in range(n_classes)
            ]
            out[strategy][str(b)] = {
                "top_ece": top_e,
                "top_nonempty_bins": int(sum(1 for r in top_rows if r["n"] > 0)),
                "mean_class_ece": float(np.mean(class_e)),
            }
    return out
