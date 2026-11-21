import numpy as np
import pytest

from rng_sequence_bench import (
    forecast_diagnostics,
    sequence_diagnostics,
    standardized_benchmark,
)


def test_brier_identity_and_fractional_ties():
    p = np.full((2, 3, 3), 1 / 3)
    y = np.array([[0, 1, 2], [2, 1, 0]])
    result = forecast_diagnostics(p, y)
    assert result["scores"] == pytest.approx([np.log(3), 2 / 3, 1 / 3])
    assert result["top_ece"] == pytest.approx(0)
    assert result["shared_logit_slope"] is None
    assert result["calibration_fit_status"] == "constant_forecasts"
    original = p.copy()
    p[0, 0] = [0.5, 0.3, 0.2]
    b = forecast_diagnostics(p, y)["brier"]
    assert b["raw"] == pytest.approx(
        b["reliability"]
        - b["resolution"]
        + b["uncertainty"]
        + b["within_bin_correction"]
    )
    assert np.array_equal(p[1], original[1])


def test_input_checks():
    with pytest.raises(ValueError, match="sum to one"):
        forecast_diagnostics(np.full((1, 2, 3), 0.2), np.array([[0, 1]]))
    with pytest.raises(ValueError, match="zero-based"):
        sequence_diagnostics([[1, 2]], [[0], [1]])


def test_sequence_metrics_on_known_transitions_and_custom_reference():
    x = sequence_diagnostics([[0, 1, 0], [1, 1, 1]], [[0], [2]], reference=[0.25, 0.75])
    assert x["repeat_mean"] == 0.5
    assert x["run_survival"] == pytest.approx([0.25, 0.25, 0])
    assert x["squared_distances"] == [0, 4]
    assert x["distance_distribution"] == [0.5, 0.5]
    assert x["transition"] == [[0, 1], [1 / 3, 2 / 3]]
    single = sequence_diagnostics([[0, 0]], [[0], [1]])
    assert single["run_survival"] == [1, 0, 0]
    assert single["tv_sd"] is None


def test_standardized_four_choice_benchmark_uses_a_separate_inner_window():
    choices = np.random.default_rng(11).integers(0, 4, size=(8, 34))
    results = standardized_benchmark(
        choices, [[0], [1], [2], [3]], [18, 26], [7, 8], inner_width=6
    )
    trials = results["tuning"]["RDC"]
    assert [(item["inner_origin"], item["outer_origin"]) for item in trials] == [
        (12, 18),
        (20, 26),
    ]
    assert "Markov2" in results["metrics"]
    assert results["calibration"]["uniform_IID"]["scores"][0] == pytest.approx(
        np.log(4)
    )
