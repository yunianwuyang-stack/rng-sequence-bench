"""Controlled recovery of finite-choice mechanisms using future choices."""

from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from scipy.stats import rankdata, wasserstein_distance

from .choice_models import (
    GRID,
    RD,
    RDC,
    design,
    design_features,
    fit_batch,
    predict,
    simulate,
)
from .evaluation import scores

CANDIDATES = {
    "iid": (),
    "repeat": ("repeat",),
    "rd": RD,
    "cycling": RDC,
    "bounded": (*RD, "cap14"),
    "frequency": (*RD, "global", "local10"),
    "momentum": (*RD, "direction"),
    "digit_bias": (*RDC, "bias"),
}
CENTERS = {
    "repeat": [-1.5],
    "rd": [-1.1, -0.5],
    "cycling": [0.15, -0.48, 0.67],
    "bounded": [0.15, -0.48, 0.67],
    "frequency": [-1.1, -0.4, 1.45, 1.25],
    "momentum": [-1.55, -1.25, 1.65],
    "digit_bias": [0.15, -0.48, 0.67],
}
SCALES = {
    "repeat": [0.55],
    "rd": [0.50, 0.26],
    "cycling": [0.55, 0.25, 0.26],
    "bounded": [0.55, 0.25, 0.26],
    "frequency": [0.55, 0.25, 0.55, 0.42],
    "momentum": [0.40, 0.30, 0.42],
    "digit_bias": [0.60, 0.26, 0.30],
}
GENERATORS = (
    "iid",
    "repeat",
    "cycling",
    "bounded",
    "frequency",
    "momentum",
    "digit_bias",
    "switch",
)


def draw_truth(mechanism, n, length, strength, rng):
    if mechanism == "iid":
        return np.empty((n, 0)), ()
    if mechanism == "switch":
        mode = np.logical_xor.accumulate(rng.random((n, length - 1)) < 0.06, axis=1)
        return np.where(
            mode[..., None], [1.0, 0.15, 0.88, 0.0], [-1.8, -1.25, 0.04, 1.75]
        ) * strength, (*RDC, "direction")
    mean = np.asarray(CENTERS[mechanism])
    spread = np.asarray(SCALES[mechanism])
    beta = rng.normal(mean, spread, size=(n, len(mean))) * strength
    if mechanism == "digit_bias":
        offsets = rng.normal(0, 0.85 * strength, size=(n, len(GRID)))
        beta = np.column_stack((beta, offsets - offsets.mean(axis=1, keepdims=True)))
    return beta, CANDIDATES[mechanism]


def profile(data):
    counts = (data[:, :, None] == np.arange(len(GRID))).mean(axis=1)
    tv = np.abs(counts - 1 / len(GRID)).sum(axis=1) / 2
    repeat = (data[:, 1:] == data[:, :-1]).mean(axis=1)
    return tv, repeat


def rank_correlation(true, estimated):
    ranks = np.stack((rankdata(true, axis=0), rankdata(estimated, axis=0)))
    ranks -= ranks.mean(axis=1, keepdims=True)
    length = np.sqrt(np.sum(ranks**2, axis=1).prod(axis=0))
    return np.divide(
        (ranks[0] * ranks[1]).sum(axis=0),
        length,
        out=np.zeros_like(length),
        where=length > 0,
    )


def parameter_errors(truth, estimates, covariance, names):
    errors = estimates - truth
    variance = np.diagonal(covariance, axis1=1, axis2=2).copy()
    if names[-1] == "bias":
        # Estimate the identifiable sum-zero contrasts from a symmetric penalty.
        k = len(GRID)
        projection = np.eye(k) - 1 / k
        variance[:, -k:] = np.einsum(
            "ij,njk,ik->ni", projection, covariance[:, -k:, -k:], projection
        )
        names = (*names[:-1], *(f"bias_{j}" for j in range(1, k + 1)))
    covered = np.abs(errors) <= 1.959964 * np.sqrt(variance)
    return {
        "names": names,
        "bias": errors.mean(axis=0).tolist(),
        "rmse": np.sqrt((errors**2).mean(axis=0)).tolist(),
        "rank": rank_correlation(truth, estimates).tolist(),
        "sign": (np.sign(estimates) == np.sign(truth)).mean(axis=0).tolist(),
        "local_interval_coverage": covered.mean(axis=0).tolist(),
    }


def recovery_distances(targets, candidates):
    return (
        np.array(
            [
                [
                    wasserstein_distance(obs_tv, fit_tv),
                    wasserstein_distance(obs_r, fit_r),
                    abs(np.quantile(obs_tv, 0.95) - np.quantile(fit_tv, 0.95)),
                ]
                for (obs_tv, obs_r), (fit_tv, fit_r) in zip(
                    targets, candidates, strict=True
                )
            ]
        )
        .mean(axis=0)
        .tolist()
    )


def one_replicate(job, *, comparison=False):
    mechanism, n, length, strength, rep, seed, coupled_validation = job
    rng = np.random.default_rng(seed)
    truth, true_terms = draw_truth(mechanism, n, length, strength, rng)
    data = simulate(truth, true_terms, GRID, rng.random((n, length)))
    validation = rng.random((3, n, length))
    target_uniforms = validation if coupled_validation else rng.random(validation.shape)
    targets = [profile(simulate(truth, true_terms, GRID, u)) for u in target_uniforms]
    independent = {}
    if comparison:
        independent_targets = [
            profile(simulate(truth, true_terms, GRID, u))
            for u in rng.random(validation.shape)
        ]
        oracle = recovery_distances(independent_targets, targets)
        if mechanism == "switch":
            fresh_states = [
                profile(
                    simulate(
                        draw_truth(mechanism, n, length, strength, rng)[0],
                        true_terms,
                        GRID,
                        u,
                    )
                )
                for u in rng.random(validation.shape)
            ]
            fresh = {}
    terms = set().union(*CANDIDATES.values())
    x = design_features(data, GRID, terms)
    cut = length * 7 // 10
    outcome = {}
    recovery = None
    for label, candidate_terms in CANDIDATES.items():
        if candidate_terms:
            design_matrix = design(x, candidate_terms)
            strength_penalty = np.ones(design_matrix.shape[-1])
            match = label == mechanism
            fitted = fit_batch(
                design_matrix[:, : cut - 1],
                data[:, 1:cut],
                strength_penalty,
                covariance=match,
            )
            beta, covariance = fitted if match else (fitted, None)
            if match:
                recovery = parameter_errors(truth, beta, covariance, candidate_terms)
            probabilities = predict(design_matrix[:, cut - 1 :], beta)
        else:
            beta = np.empty((n, 0))
            probabilities = np.full((n, length - cut, len(GRID)), 1 / len(GRID))
        heldout = scores(probabilities, data[:, cut:]).mean(axis=0)
        predicted = [
            profile(simulate(beta, candidate_terms, GRID, u)) for u in validation
        ]
        outcome[label] = [*heldout.tolist(), *recovery_distances(targets, predicted)]
        if comparison:
            independent[label] = recovery_distances(independent_targets, predicted)
            if mechanism == "switch":
                fresh[label] = recovery_distances(fresh_states, predicted)
    return {
        "generator": mechanism,
        "participants": n,
        "length": length,
        "strength": strength,
        "replicate": rep,
        "seed": seed,
        "parameters": recovery,
        "candidates": outcome,
        "forecast_winner": min(outcome, key=lambda model: outcome[model][0]),
        "tv_winner": min(outcome, key=lambda model: outcome[model][3]),
        "repeat_winner": min(outcome, key=lambda model: outcome[model][4]),
        **(
            {
                "validation_comparison": {
                    "independent": independent,
                    "oracle": oracle,
                    "fresh_switch": fresh if mechanism == "switch" else None,
                }
            }
            if comparison
            else {}
        ),
    }


def recovery_benchmark(
    *,
    sizes=(24, 96),
    lengths=(50, 100, 200, 250),
    strengths=(0.65, 1.35),
    replicates=12,
    seed=20261211,
    coupled_validation=True,
    workers=1,
    progress=None,
):
    """Return independent synthetic datasets, truth-aware parameter errors and scores.

    Each dataset uses a chronological 70/30 split. For each fitted model, three
    replicated populations compare free trajectories from the model and truth.
    For coupled validation the uniforms are shared between generators. In both
    designs the histories advance separately. The switch generator is
    an out-of-set stress test. Model recovery has no correct model for this row.
    """
    if (
        replicates < 1
        or workers < 1
        or min(sizes) < 3
        or min(lengths) < 10
        or min(strengths) <= 0
    ):
        raise ValueError(
            "Require positive strengths, at least 3 participants and 10 choices"
        )
    configs = list(
        product(GENERATORS, sizes, lengths, strengths, range(1, replicates + 1))
    )
    seeds = np.random.SeedSequence(seed).generate_state(len(configs), dtype=np.uint64)
    jobs = [
        (*config, int(subseed), coupled_validation)
        for config, subseed in zip(configs, seeds, strict=True)
    ]

    def collect(results):
        rows = []
        for row in results:
            rows.append(row)
            if progress is not None and len(rows) % 64 == 0:
                progress(len(rows), len(jobs))
        return rows

    if workers == 1:
        rows = collect(map(one_replicate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = collect(executor.map(one_replicate, jobs, chunksize=2))
    return {
        "seed": seed,
        "coupled_validation": coupled_validation,
        "replicates": replicates,
        "sizes": list(sizes),
        "lengths": list(lengths),
        "strengths": list(strengths),
        "simulation_pairs": 3,
        "score_order": [
            "log_loss",
            "brier",
            "accuracy",
            "tv_wasserstein",
            "repeat_wasserstein",
            "tv95_absolute_error",
        ],
        "generators": list(GENERATORS),
        "candidates": list(CANDIDATES),
        "switch_note": "Two known synthetic states with symmetric switch probability .06; omitted from fitting set.",
        "rows": rows,
    }
