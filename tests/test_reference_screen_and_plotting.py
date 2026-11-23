"""Tests for reference_band_screen, equivalence checks, and plotting helpers."""

import importlib.util

import numpy as np
import pytest

from rng_sequence_bench import (
    debiased_ece_l2,
    ece_binning_sensitivity,
    forecast_diagnostics,
    plot_forecast_vs_generation,
    plot_horizon_curve,
    plot_reliability_diagram,
    plot_sbc_ecdf_diff,
    predictive_compatibility_screen,
    reference_band_screen,
    tost_equivalence_test,
    tost_from_estimate,
)


def test_predictive_compatibility_screen_and_reference_band():
    reps_good = np.linspace(0.131, 0.139, 41)
    res = predictive_compatibility_screen(0.135, [0.127, 0.145], reps_good, compatibility_margin=0.010)
    assert res["screen_type"] == "simulation_predictive_compatibility"
    assert res["within_reference_band"] is True
    assert res["interval_overlaps_reference_band"] is True
    assert res["compatible_within_margin"] is True
    assert np.isclose(res["observed_percentile_in_replicates"], 21 / 41)
    assert reference_band_screen(0.135, [0.127, 0.145], reps_good, compatibility_margin=0.010) == res

    # Wide noisy replicates centered at observed pass median band screen but fail compatibility!
    reps_wide = np.linspace(0.080, 0.190, 41)
    res_wide = predictive_compatibility_screen(0.135, [0.127, 0.145], reps_wide, compatibility_margin=0.015)
    assert res_wide["within_reference_band"] is True
    assert res_wide["compatible_within_margin"] is False

    # Shifted replicates fail both
    reps_bad = np.linspace(0.155, 0.170, 41)
    res_bad = predictive_compatibility_screen(0.135, [0.127, 0.145], reps_bad, compatibility_margin=0.010)
    assert res_bad["within_reference_band"] is False
    assert res_bad["compatible_within_margin"] is False


def test_reference_band_screen_validation_errors():
    with pytest.raises(ValueError, match="lower bound"):
        reference_band_screen(0.1, [0.2, 0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least 2"):
        reference_band_screen(0.1, [0.0, 0.2], [0.1])
    with pytest.raises(ValueError, match="compatibility_margin"):
        reference_band_screen(0.1, [0.0, 0.2], [0.1, 0.12], compatibility_margin=0.0)


def test_tost_equivalence_and_boundary_cases():
    # 1. Exact analytical t-TOST on paired participant differences
    rng = np.random.default_rng(101)
    diffs_ok = rng.normal(0.001, 0.01, size=200)
    res_ok = tost_equivalence_test(paired_differences=diffs_ok, margin=0.01, alpha=0.05)
    assert res_ok["screen_type"] == "tost_equivalence"
    assert res_ok["ci_level"] == pytest.approx(0.90)
    assert res_ok["equivalent"] is True
    assert res_ok["p_tost"] < 0.05

    # 2. Boundary case: 90% CI upper bound sits strictly ON +margin -> NOT equivalent
    from scipy.stats import norm
    z95 = norm.ppf(0.95)
    res_bound = tost_from_estimate(0.01 - z95 * 0.005, 0.005, margin=0.01, df=None, alpha=0.05)
    assert np.isclose(res_bound["ci_equivalence"][1], 0.01)
    assert res_bound["equivalent"] is False

    # 3. Asymmetric failure: lower side OK, only upper side crosses +margin -> fails TOST
    res_one = tost_from_estimate(0.008, 0.004, margin=0.01, df=100, alpha=0.05)
    assert res_one["p_lower"] < 0.001
    assert res_one["p_upper"] > 0.05
    assert res_one["equivalent"] is False

    # 4. High MCSE compromises an otherwise tight point difference
    res_mcse = tost_from_estimate(0.001, 0.001, margin=0.01, df=500, alpha=0.05, mcse=0.008)
    assert res_mcse["mcse_excessive"] is True
    assert res_mcse["equivalent"] is False

    # 5. Bootstrap TOST: wide parameter+participant uncertainty rejects, narrow accepts
    boot_wide = rng.normal(0.0, 0.012, size=500)
    res_bw = tost_equivalence_test(bootstrap_diffs=boot_wide, margin=0.01, alpha=0.05)
    assert res_bw["equivalent"] is False

    boot_tight = rng.normal(0.001, 0.002, size=500)
    res_bt = tost_equivalence_test(bootstrap_diffs=boot_tight, margin=0.01, alpha=0.05)
    assert res_bt["equivalent"] is True


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib optional")
def test_plotting_helpers_headless():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax1 = plot_forecast_vs_generation(
        {
            "M1": {"log_loss": 1.92, "log_loss_ci95": [1.90, 1.94], "gen_median": 0.15, "gen_ci95": [0.14, 0.16]},
            "M2": {"log_loss": 1.90, "log_loss_ci95": [1.88, 1.92], "gen_median": 0.135, "gen_ci95": [0.13, 0.14]},
        },
        observed_value=0.134,
        observed_ci=[0.127, 0.145],
        title="Test",
    )
    assert ax1.get_title() == "Test"
    plt.close(ax1.figure)

    ax2 = plot_horizon_curve(
        [20, 40, 60],
        [0.25, 0.18, 0.15],
        {"M1": {"median": [0.26, 0.19, 0.16], "ci95": [[0.24, 0.17, 0.14], [0.28, 0.21, 0.18]]}},
        observed_ci=[[0.23, 0.27], [0.16, 0.20], [0.13, 0.17]],
    )
    assert len(ax2.lines) == 2
    plt.close(ax2.figure)

    rng = np.random.default_rng(7)
    probs = rng.dirichlet([2.0, 2.0, 2.0], size=(10, 20))
    targets = rng.integers(0, 3, size=(10, 20))
    fdiag = forecast_diagnostics(probs, targets)
    ax3 = plot_reliability_diagram(fdiag)
    assert len(ax3.lines) == 2
    plt.close(ax3.figure)

    ranks = rng.integers(0, 385, size=48)
    ax4 = plot_sbc_ecdf_diff(ranks, max_rank=384, n_sim=400)
    assert len(ax4.lines) == 2
    plt.close(ax4.figure)


def test_debiased_ece_l2_and_sensitivity():
    rng = np.random.default_rng(123)
    p = rng.dirichlet([2.0, 1.5, 1.0], size=(30, 50))
    u = rng.random((30, 50))
    y = np.minimum((u[..., None] > np.cumsum(p, axis=-1)).sum(axis=-1), 2)
    deb = debiased_ece_l2(p[:, :, 0].ravel(), (y.ravel() == 0).astype(float), n_bins=10, strategy="quantile")
    assert 0.0 <= deb < 0.05
    sens = ece_binning_sensitivity(p, y, bin_counts=(5, 10))
    assert "debiased_l2_quantile" in sens
    assert 0.0 <= sens["debiased_l2_quantile"]["10"]["top_debiased_ece"] <= 1.0
