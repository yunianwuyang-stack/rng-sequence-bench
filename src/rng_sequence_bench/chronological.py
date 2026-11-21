"""Leakage-safe ridge and smoothed Markov comparisons on fixed test trials."""

from itertools import pairwise

import numpy as np

from .choice_models import RD, RDC, design, design_features, fit_batch, predict
from .evaluation import calibration, interval, scores

RIDGES = (0.0005, 0.002, 0.008, 0.032, 0.128)
CONCENTRATIONS = (1.0, 9.0, 36.0)
MODEL_TERMS = {
    "RD": RD,
    "RDC": RDC,
    "linear_gap": (*RD, "linear_gap"),
    "inverse_gap": (*RD, "inverse_gap"),
    "binned_gap": (*RD, "gap2to4", "gap5to9", "gap10to19", "gap20plus"),
    "capped_gap": (*RD, "cap14"),
    "decaying_memory": (*RD, "memory4"),
    "RD_frequency": (*RD, "global", "local10"),
    "RDC_frequency": (*RDC, "global", "local10"),
}
MARKOV = ("individual_IID", "Markov1", "Markov2")


def standardized(x, origin):
    # A candidate-common shift cancels within the conditional probability.
    centered = x - x.mean(axis=2, keepdims=True)
    scale = np.sqrt((centered[:, : origin - 1] ** 2).mean(axis=(1, 2), keepdims=True))
    return centered / np.where(scale > 1e-12, scale, 1.0)


def choice_probabilities(x, targets, origin, right, ridge):
    x = standardized(x[:, : right - 1], origin)
    beta = fit_batch(x[:, : origin - 1], targets[:, 1:origin], (origin - 1) * ridge)
    return predict(x[:, origin - 1 :], beta)


def count_probabilities(y, origin, right, order, concentration, k=9):
    n = len(y)
    person = np.arange(n)[:, None]
    fixed = y[:, :origin]
    freq = np.stack([(fixed == j).sum(axis=1) for j in range(k)], axis=-1)
    marginal = (freq + 1) / (origin + k)
    width = right - origin
    if order == 0:
        return np.broadcast_to(marginal[:, None, :], (n, width, k))
    one = np.zeros((n, k, k))
    np.add.at(one, (person, fixed[:, :-1], fixed[:, 1:]), 1)
    one = (one + concentration * marginal[:, None, :]) / (
        one.sum(axis=-1, keepdims=True) + concentration
    )
    prev = y[:, origin - 1 : right - 1]
    if order == 1:
        return one[person, prev]
    two = np.zeros((n, k, k, k))
    np.add.at(two, (person, fixed[:, :-2], fixed[:, 1:-1], fixed[:, 2:]), 1)
    counts = two[person, y[:, origin - 2 : right - 2], prev]
    return (counts + concentration * one[person, prev]) / (
        counts.sum(axis=-1, keepdims=True) + concentration
    )


def run_windows(data, features, origins, widths, rng, k, inner_width):
    truth = np.concatenate(
        [data[:, o : o + w] for o, w in zip(origins, widths, strict=True)], axis=1
    )
    n, t = truth.shape
    p = {}
    choice_history = {}
    for name, terms in MODEL_TERMS.items():
        x = design(features, terms)
        outer, chosen = [], []
        for o, w in zip(origins, widths, strict=True):
            inner = o - inner_width
            loss = [
                scores(choice_probabilities(x, data, inner, o, r), data[:, inner:o])[
                    :, 0
                ].mean()
                for r in RIDGES
            ]
            selected = RIDGES[int(np.argmin(loss))]
            chosen.append(
                {
                    "outer_origin": o,
                    "inner_origin": inner,
                    "selected": selected,
                    "inner_loss": [float(value) for value in loss],
                }
            )
            outer.append(choice_probabilities(x, data, o, o + w, selected))
        p[name] = np.concatenate(outer, axis=1)
        choice_history[name] = chosen

    p["uniform_IID"] = np.full((n, t, k), 1 / k)
    for order, name in enumerate(MARKOV):
        outer, chosen = [], []
        for o, w in zip(origins, widths, strict=True):
            inner = o - inner_width
            grid = CONCENTRATIONS if order else (1.0,)
            losses = [
                scores(
                    count_probabilities(data, inner, o, order, r, k), data[:, inner:o]
                )[:, 0].mean()
                for r in grid
            ]
            selected = grid[int(np.argmin(losses))]
            chosen.append(
                {
                    "outer_origin": o,
                    "inner_origin": inner,
                    "selected": selected,
                    "inner_loss": [float(value) for value in losses],
                }
            )
            outer.append(count_probabilities(data, o, o + w, order, selected, k))
        p[name] = np.concatenate(outer, axis=1)
        choice_history[name] = chosen

    individual = {name: scores(prob, truth) for name, prob in p.items()}
    return {
        "origins": origins,
        "widths": widths,
        "tuning": choice_history,
        "metrics": {name: interval(value, rng) for name, value in individual.items()},
        "minus_RDC": {
            name: interval(value - individual["RDC"], rng)
            for name, value in individual.items()
        },
        "calibration": {name: calibration(prob, truth) for name, prob in p.items()},
    }


def standardized_benchmark(
    choices, coordinates, origins, widths, *, inner_width=25, seed=20261203
):
    """Tune on prior observations before fitting and scoring each outer window."""
    data = np.asarray(choices)
    coords = np.asarray(coordinates, dtype=float)
    origins = tuple(origins)
    widths = tuple(widths)
    if data.ndim != 2 or data.size == 0 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("Provide a nonempty matrix of zero-based integer choices")
    if (
        coords.ndim != 2
        or len(coords) < 2
        or coords.shape[1] < 1
        or not np.isfinite(coords).all()
    ):
        raise ValueError(
            "Provide a finite coordinates matrix with one row per category"
        )
    if data.min() < 0 or data.max() >= len(coords):
        raise ValueError("Choices fall outside the supplied alphabet")
    if not origins or len(origins) != len(widths) or inner_width < 1:
        raise ValueError(
            "Specify matching outer origins and widths with a positive inner width"
        )
    if (
        min(origins) - inner_width < 3
        or min(widths) < 1
        or max(o + w for o, w in zip(origins, widths, strict=False)) > data.shape[1]
    ):
        raise ValueError(
            "Train on at least three choices and score only recorded choices"
        )
    if any(
        a + w > b for (a, w), (b, _) in pairwise(zip(origins, widths, strict=False))
    ):
        raise ValueError("Outer test windows must be ordered and nonoverlapping")
    terms = set().union(*MODEL_TERMS.values())
    x = design_features(data, coords, terms)
    return run_windows(
        data, x, origins, widths, np.random.default_rng(seed), len(coords), inner_width
    )
