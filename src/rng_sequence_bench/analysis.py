"""Finite-sample sequence summaries and strictly chronological forecasts."""

from itertools import pairwise

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import binom

FEATURES = {
    "RD": (0, 1),
    "RDC": (0, 1, 2),
    "decaying_memory": (0, 1, 3),
    "directional_momentum": (0, 1, 4),
    "RDC_with_momentum": (0, 1, 2, 4),
}
FREQUENCY_MODELS = {
    "cumulative_frequency": (0, 1, 5),
    "local_frequency": (0, 1, 6),
    "both_frequencies": (0, 1, 5, 6),
    "RDC_with_frequencies": (0, 1, 2, 5, 6),
}


def _sequences(values, categories):
    data = np.asarray(values)
    if data.ndim != 2 or min(data.shape) < 2 or data.dtype.kind not in "iu":
        raise ValueError(
            "Expected at least two equal-length sequences of integer category indices"
        )
    if categories < 2 or data.min() < 0 or data.max() >= categories:
        raise ValueError(
            "Category indices must be between zero and categories minus one"
        )
    return data


def _variation(data, categories):
    shares = (data[..., None] == np.arange(categories)).mean(axis=-2)
    return np.abs(shares - 1 / categories).sum(axis=-1) / 2


def fixed_count_test(values, categories, permutations=999, seed=20261121):
    """Compare observed repeats with uniform reordering within each sequence."""
    data = _sequences(values, categories)
    if permutations < 1:
        raise ValueError("At least one permutation is required")
    length = data.shape[1]
    counts = (data[..., None] == np.arange(categories)).sum(axis=1)
    expected = (counts * (counts - 1)).sum(axis=1) / (length * (length - 1))
    repeat = (data[:, 1:] == data[:, :-1]).mean(axis=1)
    difference = repeat.mean() - expected.mean()
    variation = _variation(data, categories)
    count_values = np.arange(length + 1)
    iid_mean = (
        categories
        / 2
        * (
            binom.pmf(count_values, length, 1 / categories)
            * abs(count_values / length - 1 / categories)
        ).sum()
    )
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        shuffled = rng.permuted(data, axis=1)
        shuffled_difference = (
            shuffled[:, 1:] == shuffled[:, :-1]
        ).mean() - expected.mean()
        extreme += abs(shuffled_difference) >= abs(difference) - 1e-14
    controls = rng.integers(categories, size=(max(8192, len(data)), length))
    iid_variation = _variation(controls, categories)
    return {
        "participants": len(data),
        "trials": length,
        "permutations": permutations,
        "seed": seed,
        "variation_mean": float(variation.mean()),
        "variation_q95": float(np.quantile(variation, 0.95)),
        "iid_variation_mean": float(iid_mean),
        "iid_variation_q95": float(np.quantile(iid_variation, 0.95)),
        "iid_reference_sequences": len(controls),
        "observed_repeat": float(repeat.mean()),
        "expected_repeat": float(expected.mean()),
        "difference": float(difference),
        "permutation_p": float((1 + extreme) / (1 + permutations)),
    }


def _step(previous, earlier, gap, trace, distances, directions):
    k = len(distances)
    return np.stack(
        (
            previous[:, None] == np.arange(k),
            distances[previous],
            np.log2(gap),
            trace,
            directions[earlier, previous],
        ),
        axis=-1,
    )


def _geometry(coordinates, categories):
    coordinates = np.asarray(coordinates, dtype=float)
    if (
        coordinates.ndim != 2
        or len(coordinates) != categories
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError(
            "Supply a finite position vector for each outcome in alphabet order"
        )
    if len(np.unique(coordinates, axis=0)) != categories:
        raise ValueError("Position vectors must be unique")
    steps = coordinates[None, :, :] - coordinates[:, None, :]
    distances = np.linalg.norm(steps, axis=-1)
    unit = np.divide(
        steps,
        distances[..., None],
        out=np.zeros_like(steps),
        where=distances[..., None] > 0,
    )
    return distances, np.einsum("ijd,jkd->ijk", unit, unit)


def _frequency_features(data, t, total, local, window):
    people = np.arange(len(data))
    previous = data[:, t - 1]
    total[people, previous] += 1
    local[people, previous] += 1
    if t > window:
        local[people, data[:, t - window - 1]] -= 1
    k = total.shape[1]
    local_length = min(t, window)
    return np.stack(
        (
            (t / k - total) / np.sqrt(t),
            (local_length / k - local) / np.sqrt(local_length),
        ),
        axis=-1,
    )


def _feature_array(data, distances, directions, half_life, frequency_window=None):
    n, length = data.shape
    k = len(distances)
    people = np.arange(n)
    gap = np.ones((n, k), dtype=int)
    trace = np.zeros((n, k), dtype=float)
    result = np.empty((n, length - 1, k, 7 if frequency_window else 5))
    total, local = np.zeros_like(gap), np.zeros_like(gap)
    decay = 2 ** (-1 / half_life)
    for t in range(1, length):
        previous = data[:, t - 1]
        earlier = data[:, max(t - 2, 0)]
        gap += 1
        gap[people, previous] = 1
        trace *= decay
        trace[people, previous] += 1
        result[:, t - 1, :, :5] = _step(
            previous, earlier, gap, trace, distances, directions
        )
        if frequency_window is not None:
            result[:, t - 1, :, 5:] = _frequency_features(
                data, t, total, local, frequency_window
            )
    return result


def _fit(design, observations, ridge, center=None):
    observed = design[np.arange(len(observations)), observations].sum(axis=0)
    if center is None:
        center = np.zeros(design.shape[-1])

    def objective(beta):
        logits = design @ beta
        normalizer = logsumexp(logits, axis=1)
        probabilities = np.exp(logits - normalizer[:, None])
        deviation = beta - center
        loss = normalizer.sum() - observed @ beta + ridge * (deviation @ deviation) / 2
        gradient = (
            np.einsum("tk,tkd->d", probabilities, design) - observed + ridge * deviation
        )
        return loss / len(observations), gradient / len(observations)

    fit = minimize(
        objective,
        center,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-8},
    )
    if np.isfinite(fit.x).all() and np.max(np.abs(fit.jac)) > 2e-6:
        # Retry a relative-loss stop with a full-memory optimizer. The score
        # tolerance for accepting a fit is the same for both optimizers.
        fit = minimize(
            objective, fit.x, jac=True, method="BFGS", options={"gtol": 1e-8}
        )
    if not np.isfinite(fit.x).all() or np.max(np.abs(fit.jac)) > 2e-6:
        raise RuntimeError(
            f"Penalized likelihood score did not converge: {fit.message}; score={fit.jac}"
        )
    return fit.x


def _scores(probs, choices, fixed_bins):
    picked = np.take_along_axis(probs, choices[..., None], axis=-1)[..., 0]
    confidence = probs.max(axis=-1)
    ties = np.isclose(probs, confidence[..., None], atol=1e-12, rtol=0).sum(axis=-1)
    correct = np.isclose(picked, confidence, atol=1e-12, rtol=0) / ties
    brier = (probs**2).sum(axis=-1) - 2 * picked + 1
    scores = np.stack((-np.log(picked), brier, correct), axis=-1).mean(axis=1)
    bins = np.minimum((confidence * fixed_bins).astype(int), fixed_bins - 1)
    reliability = np.stack(
        [
            np.bincount(bins.ravel(), weights=weights.ravel(), minlength=fixed_bins)
            for weights in (np.ones_like(confidence), confidence, correct)
        ],
        axis=-1,
    )
    return scores, reliability


def _intervals(values, rng, reps):
    draws = np.concatenate(
        [
            values[
                rng.integers(len(values), size=(min(128, reps - start), len(values)))
            ].mean(axis=1)
            for start in range(0, reps, 128)
        ]
    )
    return {
        "mean": values.mean(axis=0).tolist(),
        "ci95": np.quantile(draws, [0.025, 0.975], axis=0).tolist(),
    }


def _summarize_bins(table):
    rows = []
    for i, (n, confidence, hits) in enumerate(table):
        rows.append(
            {
                "lower": i / len(table),
                "upper": (i + 1) / len(table),
                "n": int(n),
                "confidence": float(confidence / n) if n else None,
                "top_accuracy": float(hits / n) if n else None,
            }
        )
    ece = np.abs(table[:, 1] - table[:, 2]).sum() / table[:, 0].sum()
    return {"ece": float(ece), "bins": rows}


def _simulate(
    beta, selected, length, distances, directions, half_life, rng, frequency_window=None
):
    n = len(beta)
    k = len(distances)
    result = np.empty((n, length), dtype=int)
    result[:, 0] = rng.integers(k, size=n)
    gap = np.ones((n, k), dtype=int)
    trace = np.zeros((n, k))
    total, local = np.zeros_like(gap), np.zeros_like(gap)
    decay = 2 ** (-1 / half_life)
    people = np.arange(n)
    for t in range(1, length):
        previous = result[:, t - 1]
        gap += 1
        gap[people, previous] = 1
        trace *= decay
        trace[people, previous] += 1
        design = _step(
            previous, result[:, max(t - 2, 0)], gap, trace, distances, directions
        )
        if frequency_window is not None and max(selected) >= 5:
            frequency = _frequency_features(result, t, total, local, frequency_window)
            design = np.concatenate((design, frequency), axis=-1)
        p = softmax(np.einsum("nkd,nd->nk", design[..., selected], beta), axis=-1)
        cumulative = p.cumsum(axis=-1)
        cumulative[:, -1] = 1
        result[:, t] = (rng.random(n)[:, None] >= cumulative).sum(axis=1)
    return result


def benchmark(
    values,
    coordinates,
    origins,
    width,
    *,
    ridge=1.0,
    half_life=4.0,
    bootstrap=2000,
    simulations=40,
    seed=20261121,
    frequency_window=None,
    include_pooled=False,
):
    """Fit on prefixes, score disjoint future windows, and simulate from training fits.

    Choice indices must match the coordinate row order. Fits are participant-specific.
    Observed history advances within a scoring window while coefficients stay fixed.
    """
    k = len(coordinates)
    data = _sequences(values, k)
    origins = tuple(origins)
    n, length = data.shape
    if (
        not origins
        or min(origins) < 3
        or width < 1
        or max(origins) + width > length
        or any(b - a < width for a, b in pairwise(origins))
    ):
        raise ValueError(
            "Training prefixes and disjoint scoring windows must fit within each sequence"
        )
    if (
        not np.isfinite([ridge, half_life]).all()
        or ridge <= 0
        or half_life <= 0
        or bootstrap < 1
        or simulations < 0
    ):
        raise ValueError(
            "Use finite positive ridge, memory half-life and bootstrap count"
        )
    if frequency_window is not None and (
        not isinstance(frequency_window, (int, np.integer)) or frequency_window < 1
    ):
        raise ValueError("The frequency window must be a positive integer")
    distances, direction = _geometry(coordinates, k)
    design = _feature_array(data, distances, direction, half_life, frequency_window)
    features = FEATURES if frequency_window is None else FEATURES | FREQUENCY_MODELS
    names = ("uniform", "participant_iid", "participant_markov", *features)
    if include_pooled:
        names += ("pooled_RD", "pooled_RDC")
    scores = {name: [] for name in names}
    bins = {name: np.zeros((10, 3)) for name in names}
    rolling = []
    decomposition = []
    for origin in origins:
        train, test = data[:, :origin], data[:, origin : origin + width]
        previous = data[:, origin - 1 : origin + width - 1]
        counts = (train[..., None] == np.arange(k)).sum(axis=1) + 1
        transitions = np.ones((n, k, k))
        np.add.at(transitions, (np.arange(n)[:, None], train[:, :-1], train[:, 1:]), 1)
        transitions /= transitions.sum(axis=-1, keepdims=True)
        predicted = {
            "uniform": np.full((n, width, k), 1 / k),
            "participant_iid": np.broadcast_to(
                (counts / counts.sum(axis=-1, keepdims=True))[:, None], (n, width, k)
            ),
            "participant_markov": transitions[np.arange(n)[:, None], previous],
        }
        fits = {}
        for name, selected in features.items():
            train_x = design[:, : origin - 1, :, selected]
            test_x = design[:, origin - 1 : origin + width - 1, :, selected]
            beta = np.array(
                [_fit(x, y, ridge) for x, y in zip(train_x, train[:, 1:], strict=True)]
            )
            fits[name] = beta
            predicted[name] = softmax(np.einsum("ntkd,nd->ntk", test_x, beta), axis=-1)
        row = {
            "train_end": origin,
            "test_start": origin + 1,
            "test_end": origin + width,
        }
        pooled_x = design[:, : origin - 1, :, :3].reshape(-1, k, 3)
        pooled_y = train[:, 1:].ravel()
        pooled_rd = _fit(pooled_x[:, :, :2], pooled_y, ridge)
        pooled_rdc = _fit(pooled_x, pooled_y, ridge)
        pooled_test = design[:, origin - 1 : origin + width - 1, :, :3]
        pooled_probs = softmax(pooled_test @ pooled_rdc, axis=-1)
        if include_pooled:
            predicted["pooled_RD"] = softmax(pooled_test[..., :2] @ pooled_rd, axis=-1)
            predicted["pooled_RDC"] = pooled_probs
        for name, probs in predicted.items():
            values, reliability = _scores(probs, test, 10)
            scores[name].append(values)
            bins[name] += reliability
            row[name] = values.mean(axis=0).tolist()
        rolling.append(row)
        repeat = (test == previous).mean()
        repeat_probs = [
            np.take_along_axis(predicted[m], previous[..., None], axis=-1).mean()
            for m in ("RD", "RDC")
        ]
        centered = pooled_x - pooled_x.mean(axis=1, keepdims=True)
        zero_repeat = pooled_rdc.copy()
        zero_repeat[0] = 0
        zero_probs = softmax(pooled_test @ zero_repeat, axis=-1)
        log_gap = pooled_test[..., 2]
        repeated_candidate = pooled_test[..., 0].astype(bool)
        decomposition.append(
            {
                "train_end": origin,
                "observed_repeat": float(repeat),
                "participant_RD_repeat": float(repeat_probs[0]),
                "participant_RDC_repeat": float(repeat_probs[1]),
                "pooled_RD_repeat_coefficient": float(pooled_rd[0]),
                "pooled_RDC_repeat_coefficient": float(pooled_rdc[0]),
                "pooled_RDC_repeat_probability": float(
                    np.take_along_axis(
                        pooled_probs, previous[..., None], axis=-1
                    ).mean()
                ),
                "pooled_RDC_zero_repeat_probability": float(
                    np.take_along_axis(zero_probs, previous[..., None], axis=-1).mean()
                ),
                "correlation": np.corrcoef(
                    centered.reshape(-1, 3), rowvar=False
                ).tolist()
                if k > 2
                else None,
                "repeat_log_gap": float(log_gap[repeated_candidate].mean()),
                "nonrepeat_log_gap": float(log_gap[~repeated_candidate].mean()),
            }
        )

    rng = np.random.default_rng(seed)
    per_person = {key: np.mean(blocks, axis=0) for key, blocks in scores.items()}
    summaries = {
        key: _intervals(values, rng, bootstrap) for key, values in per_person.items()
    }
    contrasts = {
        name: _intervals(per_person[name] - per_person["RDC"], rng, bootstrap)
        for name in names
        if name != "RDC"
    }
    frequency_gap_contrast = None
    if frequency_window is not None:
        frequency_gap_contrast = _intervals(
            per_person["both_frequencies"] - per_person["RDC_with_frequencies"],
            rng,
            bootstrap,
        )
    simulation_summary = {}
    observed_tail = _variation(data, k)
    tail_threshold = np.quantile(observed_tail, 0.95)
    for name, beta in fits.items():
        statistics = []
        for _ in range(simulations):
            draw = _simulate(
                beta,
                features[name],
                length,
                distances,
                direction,
                half_life,
                rng,
                frequency_window,
            )
            tv = _variation(draw, k)
            statistics.append(
                [
                    tv.mean(),
                    np.quantile(tv, 0.95),
                    (tv >= tail_threshold).mean(),
                    (draw[:, 1:] == draw[:, :-1]).mean(),
                ]
            )
        if statistics:
            simulation_summary[name] = {
                "mean": np.mean(statistics, axis=0).tolist(),
                "simulation_q025_q975": np.quantile(
                    statistics, [0.025, 0.975], axis=0
                ).tolist(),
            }
    return {
        "participants": n,
        "trials": length,
        "seed": seed,
        "ridge": ridge,
        "bootstrap": bootstrap,
        "memory_half_life": half_life,
        "frequency_window": frequency_window,
        "paired_frequencies_without_minus_with_gap": frequency_gap_contrast,
        "score_order": [
            "log_loss_nats",
            "multiclass_brier_sum",
            "tie_aware_top_accuracy",
        ],
        "rolling": rolling,
        "participant_bootstrap": summaries,
        "paired_model_minus_RDC": contrasts,
        "calibration": {name: _summarize_bins(table) for name, table in bins.items()},
        "repeat_coefficient_decomposition": decomposition,
        "simulation_stat_order": [
            "variation_mean",
            "variation_q95",
            "share_at_or_above_observed_q95",
            "adjacent_repeat",
        ],
        "simulation_fit_train_end": origins[-1],
        "simulation_replicates": simulations,
        "simulation_observed": [
            float(observed_tail.mean()),
            float(tail_threshold),
            float((observed_tail >= tail_threshold).mean()),
            float((data[:, 1:] == data[:, :-1]).mean()),
        ],
        "plug_in_simulation": simulation_summary,
    }


def transfer_benchmark(
    training,
    evaluation,
    coordinates,
    start,
    *,
    ridge=1.0,
    bootstrap=2000,
    seed=20261121,
):
    """Fit RD and RDC to one session and forecast another session for the same rows.

    The evaluation histories are initialized in the new session. Only scores
    from the zero-based position `start` onward enter the summary.
    """
    k = len(coordinates)
    training = _sequences(training, k)
    evaluation = _sequences(evaluation, k)
    if (
        len(training) != len(evaluation)
        or not 1 <= start < evaluation.shape[1]
        or not np.isfinite(ridge)
        or ridge <= 0
        or bootstrap < 1
    ):
        raise ValueError(
            "Expected aligned participant rows, a valid scoring start and positive fitting options"
        )
    geometry = _geometry(coordinates, k)
    train_x = _feature_array(training, *geometry, 4)
    eval_x = _feature_array(evaluation, *geometry, 4)[:, start - 1 :]
    rng = np.random.default_rng(seed)
    metrics, summaries, calibration = {}, {}, {}
    for name in ("RD", "RDC"):
        selected = FEATURES[name]
        beta = np.array(
            [
                _fit(x, y, ridge)
                for x, y in zip(train_x[..., selected], training[:, 1:], strict=True)
            ]
        )
        probabilities = softmax(
            np.einsum("ntkd,nd->ntk", eval_x[..., selected], beta), axis=-1
        )
        values, reliability = _scores(probabilities, evaluation[:, start:], 10)
        metrics[name] = values
        summaries[name] = _intervals(values, rng, bootstrap)
        calibration[name] = _summarize_bins(reliability)
    return {
        "participants": len(training),
        "training_trials": training.shape[1],
        "score_start": start + 1,
        "score_end": evaluation.shape[1],
        "models": summaries,
        "calibration": calibration,
        "paired_RD_minus_RDC": _intervals(
            metrics["RD"] - metrics["RDC"], rng, bootstrap
        ),
    }


def pooling_benchmark(
    values, coordinates, origins, width, *, folds=5, precision=10.0, seed=20261121
):
    """Compare unadapted and partially pooled forecasts for participants held out of group fitting."""
    k = len(coordinates)
    data = _sequences(values, k)
    n, length = data.shape
    origins = tuple(origins)
    if (
        not 2 <= folds <= n
        or not np.isfinite(precision)
        or precision <= 0
        or not origins
        or min(origins) < 3
        or width < 1
        or max(origins) + width > length
        or any(b - a < width for a, b in pairwise(origins))
    ):
        raise ValueError(
            "Invalid participant folds, pooling precision or forecast windows"
        )
    rng = np.random.default_rng(seed)
    split = np.array_split(rng.permutation(n), folds)
    design = _feature_array(data, *_geometry(coordinates, k), 4)[:, :, :, :3]
    names = ("heldout_pooled_RD", "heldout_pooled_RDC", "heldout_partial_RDC")
    summaries = {name: np.zeros((n, 3)) for name in names}
    tables = {name: np.zeros((10, 3)) for name in names}
    for held in split:
        train_people = np.ones(n, dtype=bool)
        train_people[held] = False
        for origin in origins:
            source = design[train_people, : origin - 1].reshape(-1, k, 3)
            targets = data[train_people, 1:origin].ravel()
            rd = _fit(source[..., :2], targets, 1)
            rdc = _fit(source, targets, 1)
            prefix = design[held, : origin - 1]
            beta = np.array(
                [
                    _fit(x, y, precision, center=rdc)
                    for x, y in zip(prefix, data[held, 1:origin], strict=True)
                ]
            )
            future = design[held, origin - 1 : origin + width - 1]
            logits = (
                future[..., :2] @ rd,
                future @ rdc,
                np.einsum("ntkd,nd->ntk", future, beta),
            )
            for name, logit in zip(names, logits, strict=True):
                scores, reliability = _scores(
                    softmax(logit, axis=-1), data[held, origin : origin + width], 10
                )
                summaries[name][held] += scores / len(origins)
                tables[name] += reliability
    return {
        "participants": n,
        "folds": [len(held) for held in split],
        "origins": origins,
        "width": width,
        "partial_pooling_precision": precision,
        "fold_results": [
            {
                "fold": fold,
                "test_participants": len(held),
                "group_training_participants": n - len(held),
                "pooled_RD": float(summaries[names[0]][held, 0].mean()),
                "pooled_RDC": float(summaries[names[1]][held, 0].mean()),
                "adapted_RDC": float(summaries[names[2]][held, 0].mean()),
                "pooled_RD_minus_RDC": float(
                    (summaries[names[0]][held, 0] - summaries[names[1]][held, 0]).mean()
                ),
            }
            for fold, held in enumerate(split, start=1)
        ],
        "bootstrap": 2000,
        "models": {
            name: _intervals(scores, rng, 2000) for name, scores in summaries.items()
        },
        "calibration": {name: _summarize_bins(table) for name, table in tables.items()},
        "paired_pooled_RD_minus_RDC": _intervals(
            summaries[names[0]] - summaries[names[1]], rng, 2000
        ),
        "paired_pooled_minus_partial_RDC": _intervals(
            summaries[names[1]] - summaries[names[2]], rng, 2000
        ),
    }
