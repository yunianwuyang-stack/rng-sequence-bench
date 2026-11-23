"""Diagnostics for observed or freely generated categorical trajectories."""

import numpy as np
from scipy.special import xlogy

TV_QUANTILES = (0.50, 0.90, 0.95, 0.99)


def _participant_conditional_entropy(earlier, later, valid_pair, k):
    entropies = []
    for i in range(len(earlier)):
        vp = valid_pair[i]
        if not vp.any():
            continue
        e_i, l_i = earlier[i, vp], later[i, vp]
        trans = np.bincount(e_i * k + l_i, minlength=k * k).reshape(k, k)
        totals = trans.sum(axis=1, keepdims=True)
        probs = np.divide(trans, totals, out=np.zeros((k, k)), where=totals > 0)
        h_rows = -xlogy(probs, probs).sum(axis=1) / np.log(2)
        entropies.append(float(h_rows @ totals[:, 0] / trans.sum()))
    return np.asarray(entropies, dtype=float)


def trajectory_summary(
    choices, coordinates=None, reference=None, *, categories=None, mask=None
):
    """Summarize trajectories indexed by zero-based choices (equal-length or masked).

    Distances use Euclidean geometry. Both pooled and participant-level first-order
    conditional entropies are reported in bits. The optional ``reference`` is a
    categorical distribution defining the target for total variation. Pass an
    optional boolean ``mask`` of shape ``(participants, trials)`` to handle
    unequal sequence lengths or missing responses.
    """
    sequences = np.asarray(choices)
    if sequences.ndim != 2 or sequences.shape[0] < 1 or sequences.shape[1] < 2:
        raise ValueError("Provide at least one sequence with at least two trials")
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != sequences.shape or m.dtype != bool:
            raise ValueError("Mask must be a boolean array matching choices shape")
        if not (m.sum(axis=1) >= 2).all():
            raise ValueError("Each participant must have at least two unmasked trials")
    else:
        m = None
    coords = None if coordinates is None else np.asarray(coordinates, dtype=float)
    if coords is not None and (
        coords.ndim != 2 or not coords.shape[1] or not np.isfinite(coords).all()
    ):
        raise ValueError(
            "Provide a finite coordinates matrix with one row per category"
        )
    valid_vals = sequences if m is None else sequences[m]
    if categories is None:
        categories = (
            len(coords)
            if coords is not None
            else (
                len(reference)
                if reference is not None
                else int(valid_vals.max()) + 1
            )
        )
    if (
        not isinstance(categories, (int, np.integer))
        or categories < 2
        or (coords is not None and len(coords) != categories)
    ):
        raise ValueError("Specify the size of the category alphabet")
    k = int(categories)
    if (
        not np.issubdtype(sequences.dtype, np.integer)
        or valid_vals.min() < 0
        or valid_vals.max() >= k
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

    if m is None:
        proportions = (sequences[:, :, None] == np.arange(k)).mean(axis=1)
        valid_pair = np.ones((sequences.shape[0], sequences.shape[1] - 1), dtype=bool)
        valid_total = sequences.size
    else:
        counts = ((sequences[:, :, None] == np.arange(k)) & m[:, :, None]).sum(axis=1)
        proportions = counts / m.sum(axis=1, keepdims=True)
        valid_pair = m[:, :-1] & m[:, 1:]
        if not valid_pair.any(axis=1).all():
            raise ValueError(
                "Each participant must have at least one adjacent unmasked pair"
            )
        valid_total = int(m.sum())

    tv = np.abs(proportions - base).sum(axis=1) / 2
    earlier, later = sequences[:, :-1], sequences[:, 1:]
    repeat = (earlier == later) & valid_pair
    per_person_repeat = repeat.sum(axis=1) / valid_pair.sum(axis=1)

    # Maximal runs break at category changes or masked gaps.
    run_count = valid_total - int(repeat.sum())
    starts = repeat.copy()
    starts[:, 1:] &= ~repeat[:, :-1]
    runs = [float(starts.sum() / run_count)] if run_count > 0 else [0.0]
    for lag in (1, 2):
        starts_longer = starts[:, : max(0, starts.shape[1] - lag)].copy()
        for distance in range(1, lag + 1):
            starts_longer &= repeat[:, distance : distance + starts_longer.shape[1]]
        runs.append(float(starts_longer.sum() / run_count) if run_count > 0 else 0.0)

    e_valid = earlier[valid_pair]
    l_valid = later[valid_pair]
    transition = np.bincount(e_valid * k + l_valid, minlength=k * k).reshape(k, k)
    totals = transition.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        transition, totals, out=np.zeros((k, k)), where=totals > 0
    )
    entropy = -xlogy(probabilities, probabilities).sum(axis=1) / np.log(2)
    person_h = _participant_conditional_entropy(earlier, later, valid_pair, k)
    q25, q50, q75 = np.quantile(person_h, [0.25, 0.50, 0.75])

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
        "participant_conditional_entropy_median": float(q50),
        "participant_conditional_entropy_iqr": [float(q25), float(q75)],
        "transition": probabilities.tolist(),
    }
    if coords is not None:
        squared = np.round(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1), 12)
        levels, inverse = np.unique(squared, return_inverse=True)
        distances = np.bincount(
            inverse.reshape(k, k)[e_valid, l_valid], minlength=len(levels)
        )
        summary["squared_distances"] = levels.tolist()
        summary["distance_distribution"] = (distances / transition.sum()).tolist()
    return summary, tv


def sequence_diagnostics(
    choices, coordinates=None, reference=None, *, categories=None, mask=None
):
    """Return aggregate trajectory summaries without participant identifiers."""
    return trajectory_summary(
        choices, coordinates, reference, categories=categories, mask=mask
    )[0]
