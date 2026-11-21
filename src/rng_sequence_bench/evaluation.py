"""Participant-level forecast summaries and calibration diagnostics."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def scores(prob, target):
    actual = np.take_along_axis(prob, target[..., None], axis=-1)[..., 0]
    maximum = prob.max(axis=-1)
    ties = np.isclose(prob, maximum[..., None], rtol=0, atol=1e-12).sum(axis=-1)
    hit = np.isclose(maximum, actual, rtol=0, atol=1e-12) / ties
    return np.stack(
        (
            -np.log(np.maximum(actual, 1e-300)),
            (prob**2).sum(axis=-1) - 2 * actual + 1,
            hit,
        ),
        axis=-1,
    ).mean(axis=1)


def interval(values, rng, reps=3000):
    draws = np.concatenate(
        [
            values[
                rng.integers(len(values), size=(min(100, reps - start), len(values)))
            ].mean(axis=1)
            for start in range(0, reps, 100)
        ]
    )
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "ci95": np.quantile(draws, [0.025, 0.975], axis=0).tolist(),
    }


def reliability(probabilities, outcomes):
    group = np.minimum((probabilities * 10).astype(int), 9)
    n = np.bincount(group, minlength=10)
    predicted = np.bincount(group, weights=probabilities, minlength=10)
    actual = np.bincount(group, weights=outcomes, minlength=10)
    rows = [
        {
            "bin": i,
            "n": int(n[i]),
            "predicted": float(predicted[i] / n[i]) if n[i] else None,
            "observed": float(actual[i] / n[i]) if n[i] else None,
        }
        for i in range(10)
    ]
    return rows, float(np.abs(predicted - actual).sum() / len(probabilities))


def calibration(prob, target):
    n_classes = prob.shape[-1]
    prob = prob.reshape(-1, n_classes)
    target = target.ravel()
    n = len(target)
    events = np.eye(n_classes)[target]
    one = scores(prob[None], target[None])[0]
    category_error = []
    category_bins = []
    rel = res = unc = coarsened = 0.0
    for c in range(n_classes):
        rows, error = reliability(prob[:, c], events[:, c])
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

    log_p = np.log(np.maximum(prob, 1e-300))

    def objective(parameters):
        slope, offsets = parameters[0], parameters[1:]
        eta = slope * log_p + offsets
        normalizer = logsumexp(eta, axis=-1)
        delta = np.exp(eta - normalizer[:, None]) - events
        loss = (normalizer - eta[np.arange(n), target]).mean()
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
    picked = prob[np.arange(n), target]
    confidence = prob.max(axis=1)
    ties = np.isclose(prob, confidence[:, None], atol=1e-12, rtol=0).sum(axis=1)
    hit = np.isclose(picked, confidence, atol=1e-12, rtol=0) / ties
    top_bins, top_ece = reliability(confidence, hit)
    return {
        "scores": one.tolist(),
        "top_ece": top_ece,
        "top_bins": top_bins,
        "mean_class_ece": float(np.mean(category_error)),
        "class_ece": category_error,
        "class_bins": category_bins,
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


def forecast_diagnostics(probabilities, targets):
    """Score fixed forecasts for complete participant-by-trial trajectories.

    Probabilities have shape (participants, trials, categories) with strictly
    positive, normalized rows. Targets are zero-based integers with shape
    (participants, trials). The diagnostic regression uses the targets and
    leaves the reported probabilities and scores unchanged.
    """
    prob = np.asarray(probabilities, dtype=float)
    target = np.asarray(targets)
    if prob.ndim != 3 or min(prob.shape) == 0 or prob.shape[-1] < 2:
        raise ValueError(
            "Expected nonempty participant-by-trial-by-category probabilities"
        )
    if target.shape != prob.shape[:2] or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("Targets must be integer participant-by-trial indices")
    if target.min() < 0 or target.max() >= prob.shape[-1]:
        raise ValueError("Targets must be zero-based indices within the alphabet")
    if (
        not np.isfinite(prob).all()
        or (prob <= 0).any()
        or not np.allclose(prob.sum(axis=-1), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("Forecasts must be finite, positive and sum to one")
    return calibration(prob, target)
