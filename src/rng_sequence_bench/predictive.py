"""Posterior-predictive summaries for complete matched populations of sequences."""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .sequence_metrics import evaluate_custom_stats, trajectory_summary

GAP_EDGES = (1, 2, 4, 8, 16, 32, 64)


def recurrence_distribution(sequences, category_count, mask=None):
    """Pool intervals between recorded occurrences of the same category per person."""
    n, length = sequences.shape
    if mask is None:
        keys = (np.arange(n)[:, None] * category_count + sequences) * length
        entries = np.sort((keys + np.arange(length)).ravel())
        diffs = np.diff(entries)[entries[1:] // length == entries[:-1] // length]
    else:
        diffs_list = []
        for i in range(n):
            valid_idx = np.flatnonzero(mask[i])
            if len(valid_idx) < 2:
                continue
            vals = sequences[i, valid_idx]
            keys = vals * length + valid_idx
            entries = np.sort(keys)
            same_cat = entries[1:] // length == entries[:-1] // length
            diffs_list.append(np.diff(entries)[same_cat])
        diffs = np.concatenate(diffs_list) if diffs_list else np.array([], dtype=int)
    counts = np.histogram(diffs, [*GAP_EDGES, np.inf])[0]
    return (counts / counts.sum()).tolist() if counts.sum() else None


def predictive_profile(sequences, coordinates=None, *, categories=None, mask=None):
    aggregate, tv = trajectory_summary(
        sequences, coordinates, categories=categories, mask=mask
    )
    k = len(aggregate["transition"])
    if mask is None:
        valid_pair = np.ones((sequences.shape[0], sequences.shape[1] - 1), dtype=bool)
        e_valid, l_valid = sequences[:, :-1].ravel(), sequences[:, 1:].ravel()
        proportions = (sequences[:, :, None] == np.arange(k)).mean(axis=1)
    else:
        m = np.asarray(mask, dtype=bool)
        valid_pair = m[:, :-1] & m[:, 1:]
        e_valid = sequences[:, :-1][valid_pair]
        l_valid = sequences[:, 1:][valid_pair]
        counts = ((sequences[:, :, None] == np.arange(k)) & m[:, :, None]).sum(axis=1)
        proportions = counts / m.sum(axis=1, keepdims=True)
    repeat = ((sequences[:, 1:] == sequences[:, :-1]) & valid_pair).sum(axis=1) / valid_pair.sum(axis=1)
    aggregate["repeat_gap_distribution"] = recurrence_distribution(
        sequences, k, mask=mask
    )
    joint = np.bincount(e_valid * k + l_valid, minlength=k ** 2)
    aggregate["transition_joint"] = (
        (joint / joint.sum()).reshape(k, k).tolist()
    )
    aggregate["mean_digit_share"] = proportions.mean(axis=0).tolist()
    if coordinates is not None:
        aggregate["mean_nonrepeat_distance"] = conditional_distance(
            sequences, coordinates, mask=mask
        )
    return aggregate, tv, repeat


def conditional_distance(sequences, coordinates, mask=None):
    if mask is None:
        old, new = sequences[:, :-1], sequences[:, 1:]
    else:
        m = np.asarray(mask, dtype=bool)
        valid_pair = m[:, :-1] & m[:, 1:]
        old = sequences[:, :-1][valid_pair]
        new = sequences[:, 1:][valid_pair]
    distances = np.linalg.norm(
        np.asarray(coordinates)[old] - np.asarray(coordinates)[new], axis=-1
    )
    nonrepeat = old != new
    return float(distances[nonrepeat].mean()) if nonrepeat.any() else None


def posterior_predictive_checks(
    observed,
    replicated,
    coordinates=None,
    *,
    categories=None,
    mask=None,
    custom_stats=None,
):
    """Summarize comparable observed and posterior-replicated populations.

    Draws must retain the pairing and dimensions of the observed people.
    Predictive intervals treat each generated population as one draw. The
    pointwise coverage fractions describe the sampled participants, not
    independent hypothesis tests across statistics or participants. Optional
    ``custom_stats`` maps names to callables ``fn(sequences)`` or
    ``fn(sequences, mask)`` returning finite domain-specific summaries.
    """
    observed = np.asarray(observed)
    m_bool = None if mask is None else np.asarray(mask, dtype=bool)
    ref, tv_obs, r_obs = predictive_profile(
        observed, coordinates, categories=categories, mask=mask
    )
    obs_custom = evaluate_custom_stats(observed, m_bool, custom_stats)
    rep_custom = {name: [] for name in obs_custom}
    k = len(ref["transition"])
    static = {"tv_quantiles", "squared_distances", "run_lengths"}
    aggregates = {key: [] for key in ref.keys() - static}
    tv_people, repeat_people = [], []
    distances = {"tv_wasserstein": [], "repeat_wasserstein": [], "transition_js": []}
    for name, val in obs_custom.items():
        if np.asarray(val).ndim == 0:
            distances[f"custom_{name}_abs_diff"] = []
    for generated in replicated:
        simulated = np.asarray(generated)
        if simulated.shape != observed.shape:
            raise ValueError("Replicated and observed populations must match")
        sample, tv, rep = predictive_profile(
            simulated, coordinates, categories=k, mask=mask
        )
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
        if custom_stats is not None:
            sim_custom = evaluate_custom_stats(simulated, m_bool, custom_stats)
            for name, val in sim_custom.items():
                rep_custom[name].append(val)
                if f"custom_{name}_abs_diff" in distances:
                    distances[f"custom_{name}_abs_diff"].append(
                        float(abs(float(val) - float(obs_custom[name])))
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
    output = {
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
    if custom_stats is not None:
        output["custom_statistics"] = {
            name: table(rep_custom[name], obs_custom[name])
            for name in sorted(obs_custom)
        }
    return output


def mc_ranking_probability(discrepancies_a, discrepancies_b, *, target_prob=0.95):
    """Estimate P(D_A < D_B) and paired MCSE from matched simulation replicates.

    Uses the paired difference ``delta_r = D_{A,r} - D_{B,r}`` across ``R``
    replicates so shared random streams are accounted for in the variance.
    """
    a = np.asarray(discrepancies_a, dtype=float)
    b = np.asarray(discrepancies_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Provide paired 1-D discrepancy arrays with at least 2 replicates")
    diff = a - b
    mean_diff = float(diff.mean())
    mcse = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    prob_a_better = float(np.mean(diff < 0))
    from scipy.stats import norm
    normal_prob = float(norm.cdf(-mean_diff / mcse)) if mcse > 0 else float(mean_diff < 0)
    z_req = float(norm.ppf(target_prob))
    recommended_r = (
        int(np.ceil((z_req * diff.std(ddof=1) / abs(mean_diff)) ** 2))
        if abs(mean_diff) > 0 and diff.std(ddof=1) > 0
        else None
    )
    return {
        "replicates": int(len(diff)),
        "mean_difference": mean_diff,
        "paired_mcse": mcse,
        "ci95_mc": [mean_diff - 1.96 * mcse, mean_diff + 1.96 * mcse],
        "empirical_prob_a_better": prob_a_better,
        "normal_approx_prob_a_better": normal_prob,
        "recommended_replicates": recommended_r,
    }


def adequacy_set(observed_ci, model_values, *, margin=0.0):
    """Return models whose summary falls within the observed interval (+/- margin)."""
    lo, hi = float(observed_ci[0]) - float(margin), float(observed_ci[1]) + float(margin)
    return {
        "bounds": [lo, hi],
        "adequate": [name for name, val in model_values.items() if lo <= float(val) <= hi],
        "falsified": [name for name, val in model_values.items() if not (lo <= float(val) <= hi)],
    }
