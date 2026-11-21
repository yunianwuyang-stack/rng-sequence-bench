"""Model-independent evaluation of ordered forecasts and paired starts."""

from collections.abc import Iterable

import numpy as np

from .evaluation import forecast_diagnostics, interval, scores
from .predictive import posterior_predictive_checks


def _repeat_probabilities(probabilities, previous):
    return np.take_along_axis(probabilities, previous[..., None], axis=-1)[..., 0]


def audit_sequence_contexts(
    choices,
    coordinates,
    probabilities,
    replicated_pairs: Iterable,
    *,
    fit_through,
    initial_transitions=9,
    seed=0,
):
    """Compare observed-history forecasts with paired suffixes from two starts.

    Choices are integer (participants, trials) arrays in coordinate-row order.
    Probability row zero predicts choice index one from the first observed choice.
    Subsequent rows predict each next choice using preceding observed choices.
    Supply an iterable of (uniform-start, observed-prefix) replicate pairs. Each
    replicate includes only the suffix, in the same participant and category order
    as choices. Use the same parameters on both sides and a paired random stream.
    This function never fits or refits an external model; use the causality check
    separately to exercise a callable forecast with perturbed observations.
    """
    y = np.asarray(choices)
    p = np.asarray(probabilities, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    if y.ndim != 2 or not len(y) or coords.ndim != 2:
        raise ValueError("Choices and coordinates must be nonempty two-dimensional arrays")
    if p.shape != (y.shape[0], y.shape[1] - 1, len(coords)):
        raise ValueError("Provide one probability row for each choice after the first")
    if not 2 <= fit_through < y.shape[1] - 2:
        raise ValueError("The fitted prefix needs two choices and the suffix needs three")
    if not 1 <= initial_transitions < y.shape[1] - fit_through - 1:
        raise ValueError("The initial transition count must leave later suffix transitions")
    forecast = forecast_diagnostics(p[:, fit_through - 1 :], y[:, fit_through:])
    if (
        y.min() < 0
        or y.max() >= p.shape[-1]
        or not np.isfinite(p).all()
        or (p <= 0).any()
        or not np.allclose(p.sum(axis=-1), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("Invalid prefix choices or probability rows")
    rng = np.random.default_rng(seed)
    observed = y[:, 1:] == y[:, :-1]
    predicted = _repeat_probabilities(p, y[:, :-1])
    segments = {}
    for key, segment in (
        ("fit", slice(None, fit_through - 1)),
        ("holdout", slice(fit_through, None)),
    ):
        actual = observed[:, segment].mean(axis=1)
        estimate = predicted[:, segment].mean(axis=1)
        segments[key] = {
            "observed": float(actual.mean()),
            "predicted": float(estimate.mean()),
            "residual": interval(estimate - actual, rng),
        }

    reset, prefixed, early_late = [], [], []
    for a, b in replicated_pairs:
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != y[:, fit_through:].shape or b.shape != a.shape:
            raise ValueError("Both simulated starts must match the observed suffix shape")
        reset.append(a)
        prefixed.append(b)
        diff = (a[:, 1:] == a[:, :-1]).astype(float) - (b[:, 1:] == b[:, :-1])
        early_late.append(
            [diff[:, :initial_transitions].mean(), diff[:, initial_transitions:].mean()]
        )
    future = y[:, fit_through:]
    return {
        "fit_through": fit_through,
        "suffix_length": future.shape[1],
        "replicates": len(reset),
        "holdout_forecasts": forecast,
        "holdout_scores_ci": interval(scores(p[:, fit_through - 1 :], future), rng),
        "observed_history_repetition": segments,
        "uniform_start": posterior_predictive_checks(future, reset, coords),
        "observed_prefix": posterior_predictive_checks(future, prefixed, coords),
        "uniform_minus_prefix_repeat": {
            "initial_transitions": initial_transitions,
            "median": np.median(early_late, axis=0).tolist(),
            "predictive95": np.quantile(early_late, [0.025, 0.975], axis=0).tolist(),
        },
    }


def check_forecast_causality(forecast, choices, fit_through, categories):
    """Perturb the current and future responses at two holdout positions.

    The callable must accept integer choices, keep the original fit boundary,
    and return forecasts of all choices starting with choice index one. Forecast
    entries made through the changed choice must stay identical within tolerance.
    This checks those perturbations and does not inspect the estimator's internals.
    """
    y = np.asarray(choices)
    if y.ndim != 2 or not 2 <= fit_through < y.shape[1] or categories < 2:
        raise ValueError("Cannot perturb the requested holdout")
    reference = np.asarray(forecast(y))
    if reference.shape != (len(y), y.shape[1] - 1, categories):
        raise ValueError("Unexpected forecast dimensions")
    changes = sorted({fit_through, (fit_through + y.shape[1]) // 2})
    for index in changes:
        copy = y.copy()
        copy[:, index:] = (copy[:, index:] + 1) % categories
        changed = np.asarray(forecast(copy))
        if changed.shape != reference.shape or not np.allclose(
            reference[:, :index], changed[:, :index], rtol=0, atol=1e-12
        ):
            raise ValueError(f"Forecast depends on responses at or after trial {index + 1}")
    return {"altered_trials": [index + 1 for index in changes]}
