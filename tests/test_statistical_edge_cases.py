"""Comprehensive edge-case, masking, cardinality, and TOST boundary tests."""

import numpy as np
import pytest

from rng_sequence_bench import (
    audit_sequence_contexts,
    check_forecast_causality,
    forecast_diagnostics,
    from_ragged,
    mc_ranking_probability,
    posterior_predictive_checks,
    predictive_compatibility_screen,
    sequence_diagnostics,
    tost_equivalence_test,
    tost_from_estimate,
)
from rng_sequence_bench.choice_models import GRID, RDC, simulate


def test_category_counts_2_3_4_9_and_single_category_sequences():
    rng = np.random.default_rng(2026)
    for k in (2, 3, 4, 9):
        y = rng.integers(0, k, size=(6, 25))
        p = rng.dirichlet(np.ones(k), size=(6, 24))
        pairs = [(rng.integers(0, k, size=(6, 10)), rng.integers(0, k, size=(6, 10))) for _ in range(4)]
        sdiag = sequence_diagnostics(y, categories=k)
        fdiag = forecast_diagnostics(p, y[:, 1:])
        rep = audit_sequence_contexts(y, None, p, pairs, fit_through=15, initial_transitions=3, seed=7)
        assert len(sdiag["transition"]) == k
        assert np.isfinite(fdiag["named_scores"]["log_loss"])
        assert np.isfinite(rep["holdout_scores_named"]["log_loss"]["mean"])

    # Single-category observed sequences within a multi-category alphabet
    all_zero = np.zeros((3, 18), dtype=int)
    s_mono = sequence_diagnostics(all_zero, categories=3)
    assert s_mono["repeat_mean"] == pytest.approx(1.0)
    assert np.isfinite(s_mono["tv_mean"])

    # Degenerate categories=1 alphabet is explicitly rejected rather than producing NaN
    with pytest.raises(ValueError, match="alphabet"):
        sequence_diagnostics(all_zero, categories=1)


def test_extreme_class_imbalance_and_probability_boundaries():
    rng = np.random.default_rng(303)
    # 99% category 0, 1% category 1 across 20 participants x 50 trials
    y = (rng.random((20, 50)) < 0.01).astype(int)
    y[:, 0] = 0
    # Extreme near-boundary normalized probabilities (1e-12 vs 1 - 1e-12)
    p = np.zeros((20, 49, 2), dtype=float)
    p[..., 0] = 1.0 - 1e-12
    p[..., 1] = 1e-12

    fdiag = forecast_diagnostics(p, y[:, 1:])
    assert np.isfinite(fdiag["named_scores"]["log_loss"])
    assert np.isfinite(fdiag["named_scores"]["brier"])
    assert 0.0 <= fdiag["top_ece"] <= 1.0


def test_interior_missing_never_bridges_across_gap_and_matches_manual():
    # Sequence: 0, 0, NA, 1, 1 -> adjacent unmasked pairs are ONLY (0->0) and (1->1),
    # NEVER (0->1) across the masked position index 2!
    choices = np.array([[0, 0, 2, 1, 1], [1, 1, 2, 0, 0]], dtype=int)
    mask = np.array([[True, True, False, True, True], [True, True, False, True, True]], dtype=bool)
    sdiag = sequence_diagnostics(choices, categories=3, mask=mask)
    trans = np.asarray(sdiag["transition"])
    # Row 0 only has (0->0) and (1->1); Row 1 only has (1->1) and (0->0) -> NEVER bridge across NA!
    assert trans[0, 1] == pytest.approx(0.0)
    assert trans[1, 0] == pytest.approx(0.0)
    assert sdiag["repeat_mean"] == pytest.approx(1.0)

    # Equal-length vs ragged manual consistency on non-gap ragged lengths
    s0 = np.array([0, 1, 2, 1, 0, 2])
    s1 = np.array([2, 1, 0, 2])
    pad, m_rag = from_ragged([s0, s1])
    d_rag = sequence_diagnostics(pad, categories=3, mask=m_rag)
    d0 = sequence_diagnostics(s0[None, :], categories=3)
    d1 = sequence_diagnostics(s1[None, :], categories=3)
    assert d_rag["tv_mean"] == pytest.approx(0.5 * (d0["tv_mean"] + d1["tv_mean"]))
    assert d_rag["repeat_mean"] == pytest.approx(0.5 * (d0["repeat_mean"] + d1["repeat_mean"]))


def test_mc_paired_vs_independent_streams_and_zero_replicates():
    rng = np.random.default_rng(88)
    common = rng.normal(0.0, 0.02, size=60)
    da = common + 0.000
    db = common + 0.005 + rng.normal(0.0, 0.0005, size=60)
    paired_res = mc_ranking_probability(da, db, paired=True)
    indep_res = mc_ranking_probability(da, db, paired=False)
    assert paired_res["paired"] is True and indep_res["paired"] is False
    assert indep_res["paired_mcse"] is None
    # Independent estimator must NOT borrow paired variance reduction
    assert indep_res["mcse"] > paired_res["mcse"] * 3.0

    # Zero replicates / < 2 replicates rejected
    with pytest.raises(ValueError, match="predictive replicate"):
        posterior_predictive_checks(np.zeros((2, 5), dtype=int), [], categories=2)
    with pytest.raises(ValueError, match="at least 2"):
        predictive_compatibility_screen(0.1, [0.05, 0.15], [0.1], compatibility_margin=0.05)


def test_causality_leakage_seed_reproducibility_and_custom_stat_errors():
    y = np.array([[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]], dtype=int)

    def future_peak_forecast(arr):
        p = np.full((len(arr), arr.shape[1] - 1, 3), 0.1)
        for i in range(len(arr)):
            p[i, np.arange(arr.shape[1] - 1), arr[i, 1:]] = 0.8
        return p

    with pytest.raises(ValueError, match="depends on responses"):
        check_forecast_causality(future_peak_forecast, y, fit_through=3, categories=3)

    # Fixed seed deterministic simulation
    rng = np.random.default_rng(1234)
    beta = np.zeros((4, 3))
    s1 = simulate(beta, RDC, GRID, rng.random((4, 20)))
    rng2 = np.random.default_rng(1234)
    s2 = simulate(beta, RDC, GRID, rng2.random((4, 20)))
    assert np.array_equal(s1, s2)

    # Non-finite / failing custom stat rejected
    with pytest.raises(ValueError, match="non-finite"):
        sequence_diagnostics(y, categories=3, custom_stats={"nan_stat": lambda seq: float("nan")})
    with pytest.raises(RuntimeError):
        sequence_diagnostics(y, categories=3, custom_stats={"boom": lambda seq: (_ for _ in ()).throw(RuntimeError("fail"))})


def test_six_screen_boundary_modes_compatibility_vs_tost():
    # 1. Exact upper-bound equality -> TOST MUST reject (open interior (-margin, +margin))
    from scipy.stats import norm
    z95 = norm.ppf(0.95)
    se = 0.005
    margin = 0.020
    est_on_edge = margin - z95 * se
    assert tost_from_estimate(est_on_edge, se, margin=margin, df=None, alpha=0.05)["equivalent"] is False

    # 2. One side inside (-margin), one side exceeding (+margin) -> TOST MUST reject
    assert tost_from_estimate(0.017, 0.004, margin=0.020, df=80, alpha=0.05)["equivalent"] is False

    # 3. Centered wide simulation spread -> Route A band=True, compat=False; Route B depends on sampling SE
    wide_reps = np.linspace(0.05, 0.25, 101)
    scr_wide = predictive_compatibility_screen(0.15, [0.13, 0.17], wide_reps, compatibility_margin=0.03)
    assert scr_wide["within_reference_band"] is True
    assert scr_wide["compatible_within_margin"] is False

    # 4. Tight simulation spread shifted off-center -> fails band & compatibility
    shifted_reps = np.linspace(0.195, 0.205, 101)
    scr_shift = predictive_compatibility_screen(0.15, [0.13, 0.17], shifted_reps, compatibility_margin=0.03)
    assert scr_shift["within_reference_band"] is False
    assert scr_shift["compatible_within_margin"] is False

    # 5. Huge MCSE invalidates otherwise tight point diff & tight sampling SE
    res_hi_mcse = tost_from_estimate(0.001, 0.001, margin=0.02, df=400, alpha=0.05, mcse=0.015)
    assert res_hi_mcse["mcse_excessive"] is True
    assert res_hi_mcse["equivalent"] is False

    # 6. Large parameter uncertainty (bootstrap-refit draws) widens TOST CI vs narrow fixed-param sim
    rng = np.random.default_rng(77)
    narrow_fixed = np.full(100, 0.005) + rng.normal(0.0, 0.001, size=100)
    wide_refit = rng.normal(0.005, 0.015, size=300)
    scr_narrow = predictive_compatibility_screen(0.15, [0.13, 0.17], 0.15 + narrow_fixed, compatibility_margin=0.02)
    tost_refit = tost_equivalence_test(bootstrap_diffs=wide_refit, margin=0.02, alpha=0.05)
    assert scr_narrow["compatible_within_margin"] is True
    assert tost_refit["equivalent"] is False
