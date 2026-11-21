"""Diagnostics for observed or freely generated categorical trajectories."""

import numpy as np
from scipy.special import xlogy

TV_QUANTILES = (0.50, 0.90, 0.95, 0.99)


def trajectory_summary(choices, coordinates=None, reference=None, *, categories=None):
    """Summarize equal-length trajectories indexed by zero-based choices.

    Distances use Euclidean geometry. The entropy is the pooled first-order
    conditional entropy in bits. The optional reference is a categorical
    distribution defining the target for total variation.
    """
    sequences = np.asarray(choices)
    if sequences.ndim != 2 or sequences.shape[0] < 1 or sequences.shape[1] < 2:
        raise ValueError("Provide at least one sequence with at least two trials")
    coords = None if coordinates is None else np.asarray(coordinates, dtype=float)
    if coords is not None and (coords.ndim != 2 or not coords.shape[1] or not np.isfinite(coords).all()):
        raise ValueError(
            "Provide a finite coordinates matrix with one row per category"
        )
    if categories is None:
        categories = len(coords) if coords is not None else (
            len(reference) if reference is not None else int(sequences.max()) + 1
        )
    if not isinstance(categories, (int, np.integer)) or categories < 2 or (
        coords is not None and len(coords) != categories
    ):
        raise ValueError("Specify the size of the category alphabet")
    k = categories
    if (
        not np.issubdtype(sequences.dtype, np.integer)
        or sequences.min() < 0
        or sequences.max() >= k
    ):
        raise ValueError(
            "Choices must be zero-based integer indices within the alphabet"
        )
    base = (
        np.full(k, 1 / k) if reference is None else np.asarray(reference, dtype=float)
    )
    if (
        base.shape != (k,)
        or not np.isfinite(base).all()
        or (base < 0).any()
        or not np.isclose(base.sum(), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("Reference must be a normalized categorical distribution")

    proportions = (sequences[:, :, None] == np.arange(k)).mean(axis=1)
    tv = np.abs(proportions - base).sum(axis=1) / 2
    earlier, later = sequences[:, :-1], sequences[:, 1:]
    repeat = earlier == later
    per_person_repeat = repeat.mean(axis=1)
    run_count = sequences.size - repeat.sum()
    starts = repeat.copy()
    starts[:, 1:] &= ~repeat[:, :-1]
    runs = [float(starts.sum() / run_count)]
    for lag in (1, 2):
        starts_longer = starts[:, : max(0, starts.shape[1] - lag)].copy()
        for distance in range(1, lag + 1):
            starts_longer &= repeat[:, distance : distance + starts_longer.shape[1]]
        runs.append(float(starts_longer.sum() / run_count))

    transition = np.bincount((earlier * k + later).ravel(), minlength=k * k).reshape(
        k, k
    )
    totals = transition.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        transition, totals, out=np.zeros((k, k)), where=totals > 0
    )
    entropy = -xlogy(probabilities, probabilities).sum(axis=1) / np.log(2)
    summary = {
        "tv_mean": float(tv.mean()),
        "tv_sd": float(tv.std(ddof=1)) if len(tv) > 1 else None,
        "tv_quantiles": list(TV_QUANTILES),
        "tv_q": np.quantile(tv, TV_QUANTILES).tolist(),
        "repeat_mean": float(per_person_repeat.mean()),
        "repeat_sd": float(per_person_repeat.std(ddof=1)) if len(tv) > 1 else None,
        "run_lengths": [2, 3, 4],
        "run_survival": runs,
        "first_order_conditional_entropy": float(
            entropy @ totals[:, 0] / transition.sum()
        ),
        "transition": probabilities.tolist(),
    }
    if coords is not None:
        squared = np.round(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1), 12)
        levels, inverse = np.unique(squared, return_inverse=True)
        distances = np.bincount(inverse.reshape(k, k)[earlier, later].ravel(), minlength=len(levels))
        summary["squared_distances"] = levels.tolist()
        summary["distance_distribution"] = (distances / transition.sum()).tolist()
    return summary, tv


def sequence_diagnostics(choices, coordinates=None, reference=None, *, categories=None):
    """Return aggregate trajectory summaries without participant identifiers."""
    return trajectory_summary(choices, coordinates, reference, categories=categories)[0]
