"""Tests for reference_band_screen, predictive-margin checks, and plotting helpers."""

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
    reference_band_screen,
)


def test_reference_band_screen_and_predictive_margin():
    reps_good = np.linspace(0.131, 0.139, 41)
    res = reference_band_screen(0.135, [0.127, 0.145], reps_good, practical_margin=0.010)
    assert res["within_reference_band"] is True
    assert res["interval_overlaps_reference_band"] is True
    assert res["predictive_interval_within_margin"] is True
    assert res["practical_margin_status"] == "within"
    assert res["margin_screen_basis"] == "95% interval of simulated cohort-level discrepancies"
    assert "equivalent_within_margin" not in res
    assert "tost" not in str(res).lower()
    assert np.isclose(res["observed_percentile_in_replicates"], 21 / 41)

    # Wide noisy replicates centered at observed pass median band screen but fall outside the practical margin!
    reps_wide = np.linspace(0.080, 0.190, 41)
    res_wide = reference_band_screen(0.135, [0.127, 0.145], reps_wide, practical_margin=0.015)
    assert res_wide["within_reference_band"] is True
    assert res_wide["predictive_interval_within_margin"] is False
    assert res_wide["practical_margin_status"] == "indeterminate"

    # A narrow shifted replicate distribution is outside the practical margin.
    reps_outside = np.linspace(0.170, 0.178, 41)
    res_outside = reference_band_screen(
        0.135, [0.127, 0.145], reps_outside, practical_margin=0.010
    )
    assert res_outside["practical_margin_status"] == "outside"

    # Shifted replicates fail both
    reps_bad = np.linspace(0.155, 0.170, 41)
    res_bad = reference_band_screen(0.135, [0.127, 0.145], reps_bad, practical_margin=0.010)
    assert res_bad["within_reference_band"] is False
    assert res_bad["predictive_interval_within_margin"] is False


def test_reference_band_screen_validation_errors():
    with pytest.raises(ValueError, match="lower bound"):
        reference_band_screen(0.1, [0.2, 0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least 2"):
        reference_band_screen(0.1, [0.0, 0.2], [0.1])
    with pytest.raises(ValueError, match="practical_margin"):
        reference_band_screen(0.1, [0.0, 0.2], [0.1, 0.12], practical_margin=0.0)


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
