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


def mc_ranking_probability(
    discrepancies_a,
    discrepancies_b,
    *,
    target_prob=0.95,
    paired=True,
):
    """Estimate P(D_A < D_B) and MCSE from matched or independent simulations.

    When ``paired=True`` (common random streams), variance uses the paired
    replicate differences ``delta_r = D_{A,r} - D_{B,r}``. When ``paired=False``
    (independent random streams), variance uses uncoupled replicate means
    ``Var(D_A)/len(A) + Var(D_B)/len(B)`` so unpaired runs never borrow paired
    variance reduction.
    """
    a = np.asarray(discrepancies_a, dtype=float)
    b = np.asarray(discrepancies_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 2:
        raise ValueError("Provide 1-D discrepancy arrays with at least 2 replicates each")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Discrepancy arrays must be finite")
    from scipy.stats import norm

    if paired:
        if a.shape != b.shape:
            raise ValueError("Paired MC ranking requires identical replicate array shapes")
        diff = a - b
        mean_diff = float(diff.mean())
        sd_eff = float(diff.std(ddof=1))
        mcse = float(sd_eff / np.sqrt(len(diff)))
        prob_a_better = float(np.mean(diff < 0))
        replicates = int(len(diff))
    else:
        mean_diff = float(a.mean() - b.mean())
        var_sum = float(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        mcse = float(np.sqrt(var_sum))
        sd_eff = float(mcse * np.sqrt(min(len(a), len(b))))
        prob_a_better = float(norm.cdf(-mean_diff / mcse)) if mcse > 0 else float(mean_diff < 0)
        replicates = int(min(len(a), len(b)))
    normal_prob = float(norm.cdf(-mean_diff / mcse)) if mcse > 0 else float(mean_diff < 0)
    z_req = float(norm.ppf(target_prob))
    recommended_r = (
        int(np.ceil((z_req * sd_eff / abs(mean_diff)) ** 2))
        if abs(mean_diff) > 0 and sd_eff > 0
        else None
    )
    return {
        "paired": bool(paired),
        "replicates": replicates,
        "mean_difference": mean_diff,
        "mcse": mcse,
        "paired_mcse": mcse if paired else None,
        "ci95_mc": [mean_diff - 1.96 * mcse, mean_diff + 1.96 * mcse],
        "empirical_prob_a_better": prob_a_better,
        "normal_approx_prob_a_better": normal_prob,
        "recommended_replicates": recommended_r,
    }


def adequacy_set(observed_ci, model_values, *, margin=0.0):
    """Descriptive reference-band screen: models whose medians lie in observed CI.

    This is a descriptive empirical reference-band check, not a formal TOST
    equivalence test or overall multidimensional adequacy verdict.
    """
    lo, hi = float(observed_ci[0]) - float(margin), float(observed_ci[1]) + float(margin)
    if lo > hi:
        raise ValueError("observed_ci lower bound must not exceed upper bound")
    within = [name for name, val in model_values.items() if lo <= float(val) <= hi]
    outside = [name for name, val in model_values.items() if not (lo <= float(val) <= hi)]
    return {
        "screen_type": "descriptive_reference_band",
        "bounds": [lo, hi],
        "within_band": within,
        "outside_band": outside,
    }


def predictive_compatibility_screen(
    observed_value,
    observed_ci,
    replicated_values,
    *,
    compatibility_margin=None,
):
    """Route A: Descriptive reference band + simulation predictive compatibility.

    Examines two simulation-distribution properties across ``R`` generated
    populations: (1) whether the simulated median lies inside the observed
    participant-bootstrap 95% reference band (``within_reference_band``), and
    (2) whether the 95% simulation predictive discrepancy interval
    ``[Q_0.025(reps - obs), Q_0.975(reps - obs)]`` falls inside
    ``[-compatibility_margin, +compatibility_margin]``.

    Note: ``difference_ci95`` is a predictive distribution interval across
    simulated cohorts, NOT a sampling confidence interval or frequentist TOST
    on the population discrepancy parameter. Use ``tost_equivalence_test`` or
    ``tost_from_estimate`` for inferential TOST equivalence testing.
    """
    obs = float(observed_value)
    lo, hi = float(observed_ci[0]), float(observed_ci[1])
    if lo > hi:
        raise ValueError("observed_ci lower bound must not exceed upper bound")
    if not (np.isfinite(obs) and np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("Observed value and interval bounds must be finite")
    reps = np.asarray(replicated_values, dtype=float)
    if reps.ndim != 1 or len(reps) < 2 or not np.isfinite(reps).all():
        raise ValueError("Provide a finite 1-D array with at least 2 replicate values")
    med = float(np.median(reps))
    ci50 = np.quantile(reps, [0.25, 0.75]).tolist()
    ci90 = np.quantile(reps, [0.05, 0.95]).tolist()
    ci95 = np.quantile(reps, [0.025, 0.975]).tolist()
    diffs = reps - obs
    diff_ci95 = np.quantile(diffs, [0.025, 0.975]).tolist()
    obs_pct = float(np.mean(reps <= obs))
    out = {
        "screen_type": "simulation_predictive_compatibility",
        "observed": obs,
        "observed_reference_band_95": [lo, hi],
        "replicates": int(len(reps)),
        "replicate_median": med,
        "replicate_ci50": ci50,
        "replicate_ci90": ci90,
        "replicate_ci95": ci95,
        "observed_percentile_in_replicates": obs_pct,
        "difference_median": float(np.median(diffs)),
        "difference_ci95": diff_ci95,
        "within_reference_band": bool(lo <= med <= hi),
        "interval_overlaps_reference_band": bool(ci95[1] >= lo and ci95[0] <= hi),
    }
    if compatibility_margin is not None:
        delta = float(compatibility_margin)
        if delta <= 0 or not np.isfinite(delta):
            raise ValueError("compatibility_margin must be a finite positive float")
        out["compatibility_margin"] = delta
        out["compatible_within_margin"] = bool(diff_ci95[0] >= -delta and diff_ci95[1] <= delta)
    return out


def reference_band_screen(
    observed_value,
    observed_ci,
    replicated_values,
    *,
    compatibility_margin=None,
):
    """Backward-named Route A wrapper around ``predictive_compatibility_screen``."""
    return predictive_compatibility_screen(
        observed_value,
        observed_ci,
        replicated_values,
        compatibility_margin=compatibility_margin,
    )


def tost_from_estimate(
    estimate,
    se,
    *,
    margin,
    df=None,
    alpha=0.05,
    mcse=0.0,
    inference_basis="estimate_se",
):
    """Route B: Genuine frequentist Two One-Sided Tests (TOST) for equivalence.

    Tests ``H0: theta <= -margin or theta >= +margin`` against
    ``H1: -margin < theta < +margin`` at one-sided level ``alpha``,
    equivalently checking whether the ``(1 - 2*alpha)`` (90% when ``alpha=0.05``)
    inferential confidence interval lies strictly inside ``(-margin, +margin)``.

    ``se`` represents participant sampling (and, where estimated by refit,
    parameter) uncertainty; optional ``mcse`` captures finite-replicate Monte
    Carlo error and is propagated as ``effective_se = hypot(se, mcse)``.
    """
    est = float(estimate)
    se_val = float(se)
    mcse_val = float(mcse)
    delta = float(margin)
    alpha_val = float(alpha)
    if not (np.isfinite(est) and np.isfinite(se_val) and np.isfinite(mcse_val)):
        raise ValueError("estimate, se, and mcse must be finite")
    if se_val < 0 or mcse_val < 0:
        raise ValueError("se and mcse must be non-negative")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("margin must be a finite positive float")
    if not np.isfinite(alpha_val) or not (0.0 < alpha_val < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")
    if df is not None and (not np.isfinite(float(df)) or float(df) <= 0):
        raise ValueError("df must be positive when provided")

    from scipy.stats import norm, t as student_t

    eff_se = float(np.hypot(se_val, mcse_val))
    dist_obj = student_t(float(df)) if df is not None else norm
    crit = float(dist_obj.ppf(1.0 - alpha_val))

    if eff_se == 0.0:
        p_lower = 0.0 if est > -delta else 1.0
        p_upper = 0.0 if est < delta else 1.0
        ci_low, ci_high = est, est
    else:
        t_lower = (est + delta) / eff_se
        t_upper = (delta - est) / eff_se
        p_lower = float(dist_obj.sf(t_lower))
        p_upper = float(dist_obj.sf(t_upper))
        ci_low = est - crit * eff_se
        ci_high = est + crit * eff_se

    p_tost = float(max(p_lower, p_upper))
    equivalent = bool(p_tost < alpha_val and ci_low > -delta and ci_high < delta)
    mcse_ratio = float(mcse_val / delta)
    mcse_warn = bool(mcse_val > 0.10 * delta and (se_val == 0.0 or mcse_val > 0.25 * se_val))
    return {
        "screen_type": "tost_equivalence",
        "inference_basis": inference_basis,
        "estimate": est,
        "sampling_se": se_val,
        "mcse": mcse_val,
        "effective_se": eff_se,
        "df": None if df is None else float(df),
        "alpha": alpha_val,
        "ci_level": 1.0 - 2.0 * alpha_val,
        "ci_equivalence": [float(ci_low), float(ci_high)],
        "margin": delta,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": equivalent,
        "mcse_to_margin_ratio": mcse_ratio,
        "mcse_excessive": mcse_warn,
    }


def tost_equivalence_test(
    *,
    margin,
    paired_differences=None,
    bootstrap_diffs=None,
    alpha=0.05,
    mcse=0.0,
):
    """Route B: Run genuine statistical TOST from paired data or bootstrap draws.

    Exactly one of ``paired_differences`` or ``bootstrap_diffs`` must be
    supplied:
      - ``paired_differences``: length-``N`` vector of participant-level
        differences ``d_i = s_i^{sim} - s_i^{obs}`` tested via one-sample
        paired ``t``-TOST with ``df = N - 1``.
      - ``bootstrap_diffs``: length-``B`` vector of participant-bootstrap (or
        bootstrap-refit) population discrepancy draws
        ``theta_b = S(Y^{sim,(b)}) - S(Y^{obs,(b)})`` tested via ``(1-2*alpha)``
        bootstrap percentile + MCSE-propagated TOST.
    """
    if (paired_differences is None) == (bootstrap_diffs is None):
        raise ValueError("Supply exactly one of paired_differences or bootstrap_diffs")
    delta = float(margin)
    alpha_val = float(alpha)
    mcse_val = float(mcse)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("margin must be a finite positive float")
    if not np.isfinite(alpha_val) or not (0.0 < alpha_val < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")
    if not np.isfinite(mcse_val) or mcse_val < 0:
        raise ValueError("mcse must be non-negative and finite")

    if paired_differences is not None:
        diffs = np.asarray(paired_differences, dtype=float)
        if diffs.ndim != 1 or len(diffs) < 2 or not np.isfinite(diffs).all():
            raise ValueError("paired_differences must be a finite 1-D array with N >= 2")
        est = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        res = tost_from_estimate(
            est,
            se,
            margin=delta,
            df=len(diffs) - 1,
            alpha=alpha_val,
            mcse=mcse_val,
            inference_basis="paired_participant_t",
        )
        res["participants"] = int(len(diffs))
        return res

    draws = np.asarray(bootstrap_diffs, dtype=float)
    if draws.ndim != 1 or len(draws) < 20 or not np.isfinite(draws).all():
        raise ValueError("bootstrap_diffs must be a finite 1-D array with at least 20 draws")
    est = float(draws.mean())
    boot_se = float(draws.std(ddof=1))
    if mcse_val > 0.0:
        if boot_se == 0.0:
            # No participant variance observed; fall back to normal MCSE TOST interval.
            res = tost_from_estimate(
                est,
                0.0,
                margin=delta,
                df=None,
                alpha=alpha_val,
                mcse=mcse_val,
                inference_basis="bootstrap_plus_mcse",
            )
            res["bootstrap_draws"] = int(len(draws))
            res["sampling_se"] = 0.0
            return res
        scale = float(np.hypot(boot_se, mcse_val) / boot_se)
        adj_draws = est + scale * (draws - est)
    else:
        adj_draws = draws
    eff_se = float(np.hypot(boot_se, mcse_val))
    ci_lo, ci_hi = [float(x) for x in np.quantile(adj_draws, [alpha_val, 1.0 - alpha_val])]
    p_lower = float(np.mean(adj_draws <= -delta))
    p_upper = float(np.mean(adj_draws >= delta))
    p_tost = float(max(p_lower, p_upper))
    equivalent = bool(p_tost < alpha_val and ci_lo > -delta and ci_hi < delta)
    mcse_ratio = float(mcse_val / delta)
    mcse_warn = bool(mcse_val > 0.10 * delta and (boot_se == 0.0 or mcse_val > 0.25 * boot_se))
    return {
        "screen_type": "tost_equivalence",
        "inference_basis": "participant_bootstrap_percentile",
        "bootstrap_draws": int(len(draws)),
        "estimate": est,
        "sampling_se": boot_se,
        "mcse": mcse_val,
        "effective_se": eff_se,
        "alpha": alpha_val,
        "ci_level": 1.0 - 2.0 * alpha_val,
        "ci_equivalence": [ci_lo, ci_hi],
        "margin": delta,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": equivalent,
        "mcse_to_margin_ratio": mcse_ratio,
        "mcse_excessive": mcse_warn,
    }
