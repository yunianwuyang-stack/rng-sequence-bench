"""Edge cases, named outputs, calibration sensitivity, and MC utilities."""

import numpy as np

from rng_sequence_bench import (
    adequacy_set,
    audit_sequence_contexts,
    ece_binning_sensitivity,
    forecast_diagnostics,
    mc_ranking_probability,
    sequence_diagnostics,
)


def test_single_participant_and_single_observed_category():
    # Single participant choosing only category 0 out of K=3 categories
    y = np.zeros((1, 20), dtype=int)
    p = np.full((1, 19, 3), 1.0 / 3.0)
    pairs = [(np.zeros((1, 8), dtype=int), np.zeros((1, 8), dtype=int))]

    seq = sequence_diagnostics(y, categories=3)
    assert seq["tv_sd"] is None
    assert seq["repeat_mean"] == 1.0
    assert np.isfinite(seq["first_order_conditional_entropy"])

    fdiag = forecast_diagnostics(p, y[:, 1:])
    assert fdiag["calibration_fit_status"] == "missing_observed_category"
    assert np.isfinite(fdiag["named_scores"]["log_loss"])

    report = audit_sequence_contexts(y, None, p, pairs, fit_through=12, initial_transitions=2)
    assert np.isclose(report["holdout_scores_named"]["log_loss"]["mean"], np.log(3))
    assert report["holdout_scores_named"]["log_loss"]["ci95"][0] == report["holdout_scores_named"]["log_loss"]["ci95"][1]


def test_seed_reproducibility_and_named_scores_consistency():
    rng = np.random.default_rng(999)
    y = rng.integers(4, size=(10, 30))
    raw = rng.dirichlet(np.ones(4), size=(10, 29))
    pairs = [(rng.integers(4, size=(10, 10)), rng.integers(4, size=(10, 10))) for _ in range(6)]

    r1 = audit_sequence_contexts(y, None, raw, pairs, fit_through=20, initial_transitions=4, seed=2026)
    r2 = audit_sequence_contexts(y, None, raw, pairs, fit_through=20, initial_transitions=4, seed=2026)
    assert r1 == r2
    assert r1["holdout_forecasts"]["named_scores"]["log_loss"] == r1["holdout_forecasts"]["scores"][0]
    assert r1["holdout_forecasts"]["named_scores"]["brier"] == r1["holdout_forecasts"]["scores"][1]
    assert r1["holdout_forecasts"]["named_scores"]["top1_accuracy"] == r1["holdout_forecasts"]["scores"][2]


def test_ece_binning_sensitivity_and_corp():
    rng = np.random.default_rng(17)
    p = rng.dirichlet([2.0, 1.0, 1.0], size=(20, 40))
    u = rng.random((20, 40))
    y = (u[..., None] > np.cumsum(p, axis=-1)).sum(axis=-1)
    y = np.minimum(y, 2)
    sens = ece_binning_sensitivity(p, y, bin_counts=(5, 10, 15))
    assert set(sens["uniform"].keys()) == {"5", "10", "15"}
    assert set(sens["quantile"].keys()) == {"5", "10", "15"}
    assert 0.0 <= sens["corp"]["corp_ece"] <= 1.0


def test_mc_ranking_probability_and_adequacy_set():
    rng = np.random.default_rng(55)
    d_a = rng.normal(0.010, 0.002, size=50)
    d_b = rng.normal(0.015, 0.002, size=50)
    rank = mc_ranking_probability(d_a, d_b)
    assert rank["mean_difference"] < 0
    assert rank["normal_approx_prob_a_better"] > 0.99
    assert rank["recommended_replicates"] is not None

    adeq = adequacy_set([0.1267, 0.1456], {"RDC": 0.0882, "RDC+freq": 0.1567, "Cap+offsets": 0.1362})
    assert adeq["inside_reference_band"] == ["Cap+offsets"]
    assert set(adeq["outside_reference_band"]) == {"RDC", "RDC+freq"}
