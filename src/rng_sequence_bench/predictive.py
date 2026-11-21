"""Posterior-predictive summaries for complete matched populations of sequences."""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .sequence_metrics import trajectory_summary

GAP_EDGES = (1, 2, 4, 8, 16, 32, 64)


def recurrence_distribution(sequences, category_count):
    """Pool intervals between recorded occurrences of the same category per person."""
    n, length = sequences.shape
    keys = (np.arange(n)[:, None] * category_count + sequences) * length
    entries = np.sort((keys + np.arange(length)).ravel())
    diffs = np.diff(entries)[entries[1:] // length == entries[:-1] // length]
    counts = np.histogram(diffs, [*GAP_EDGES, np.inf])[0]
    return (counts / counts.sum()).tolist() if counts.sum() else None


def predictive_profile(sequences, coordinates=None, *, categories=None):
    aggregate, tv = trajectory_summary(sequences, coordinates, categories=categories)
    k = len(aggregate["transition"])
    repeat = (sequences[:, 1:] == sequences[:, :-1]).mean(axis=1)
    aggregate["repeat_gap_distribution"] = recurrence_distribution(
        sequences, k
    )
    joint = np.bincount(
        (sequences[:, :-1] * k + sequences[:, 1:]).ravel(),
        minlength=k ** 2,
    )
    aggregate["transition_joint"] = (
        (joint / joint.sum()).reshape(k, k).tolist()
    )
    proportions = (sequences[:, :, None] == np.arange(k)).mean(axis=1)
    aggregate["mean_digit_share"] = proportions.mean(axis=0).tolist()
    if coordinates is not None:
        aggregate["mean_nonrepeat_distance"] = conditional_distance(sequences, coordinates)
    return aggregate, tv, repeat


def conditional_distance(sequences, coordinates):
    old, new = sequences[:, :-1], sequences[:, 1:]
    distances = np.linalg.norm(
        np.asarray(coordinates)[old] - np.asarray(coordinates)[new], axis=-1
    )
    nonrepeat = old != new
    return float(distances[nonrepeat].mean()) if nonrepeat.any() else None


def posterior_predictive_checks(observed, replicated, coordinates=None, *, categories=None):
    """Summarize comparable observed and posterior-replicated populations.

    Draws must retain the pairing and dimensions of the observed people.
    Predictive intervals treat each generated population as one draw. The
    pointwise coverage fractions describe the sampled participants, not
    independent hypothesis tests across statistics or participants.
    """
    observed = np.asarray(observed)
    ref, tv_obs, r_obs = predictive_profile(observed, coordinates, categories=categories)
    k = len(ref["transition"])
    static = {"tv_quantiles", "squared_distances", "run_lengths"}
    aggregates = {key: [] for key in ref.keys() - static}
    tv_people, repeat_people = [], []
    distances = {"tv_wasserstein": [], "repeat_wasserstein": [], "transition_js": []}
    for generated in replicated:
        simulated = np.asarray(generated)
        if simulated.shape != observed.shape:
            raise ValueError("Replicated and observed populations must match")
        sample, tv, rep = predictive_profile(simulated, coordinates, categories=k)
        tv_people.append(tv)
        repeat_people.append(rep)
        for key, group in aggregates.items():
            if sample[key] is not None:
                group.append(sample[key])
        distances["tv_wasserstein"].append(wasserstein_distance(tv_obs, tv))
        distances["repeat_wasserstein"].append(wasserstein_distance(r_obs, rep))
        distances["transition_js"].append(
            jensenshannon(
                np.ravel(ref["transition_joint"]),
                np.ravel(sample["transition_joint"]),
                base=2,
            )
        )
    if not tv_people:
        raise ValueError("Require at least one recorded predictive replicate")

    def table(values, truth=None):
        array = np.asarray(values)
        summary = {
            "median": np.median(array, axis=0).tolist(),
            "predictive95": np.quantile(array, [0.025, 0.975], axis=0).tolist(),
            "replicates_defined": len(array),
        }
        if truth is not None:
            summary.update(
                observed=truth,
                observed_minus_median=(
                    np.asarray(truth) - np.median(array, axis=0)
                ).tolist(),
                tail_fraction=np.mean(array >= truth, axis=0).tolist(),
            )
        return summary

    def coverage(actual, sampled):
        lo, hi = np.quantile(sampled, [0.025, 0.975], axis=0)
        return float(np.mean((actual >= lo) & (actual <= hi)))

    checks = {
        key: table(values, ref[key])
        for key, values in sorted(aggregates.items())
        if values
    }
    return {
        "replicates": len(tv_people),
        "participants": len(observed),
        "length": observed.shape[1],
        "index": {
            **{key: ref[key] for key in sorted(static & ref.keys())},
            "gap_left_edges": list(GAP_EDGES),
        },
        "statistics": checks,
        "distance_from_observed": {key: table(vals) for key, vals in distances.items()},
        "same_person_coverage95": {
            "tv": coverage(tv_obs, tv_people),
            "repeat": coverage(r_obs, repeat_people),
        },
    }
