"""Input contract and validation tests for RNG Sequence Bench."""

import numpy as np
import pytest

from rng_sequence_bench import (
    audit_sequence_contexts,
    check_forecast_causality,
    forecast_diagnostics,
    read_ragged_trials,
    read_trials,
    sequence_diagnostics,
)


def _valid_audit_inputs(n=4, t=20, k=3, cut=12):
    rng = np.random.default_rng(101)
    y = rng.integers(k, size=(n, t))
    p = np.full((n, t - 1, k), 1.0 / k)
    s1 = rng.integers(k, size=(n, t - cut))
    s2 = rng.integers(k, size=(n, t - cut))
    return y, p, [(s1, s2)]


def test_rejects_negative_and_out_of_range_categories():
    y, p, pairs = _valid_audit_inputs()
    bad_neg = y.copy()
    bad_neg[0, 0] = -1
    with pytest.raises(ValueError):
        audit_sequence_contexts(bad_neg, None, p, pairs, fit_through=12)
    with pytest.raises(ValueError):
        sequence_diagnostics(bad_neg, categories=3)

    bad_hi = y.copy()
    bad_hi[0, 1] = 3
    with pytest.raises(ValueError):
        audit_sequence_contexts(bad_hi, None, p, pairs, fit_through=12)
    with pytest.raises(ValueError):
        forecast_diagnostics(p, bad_hi[:, 1:])


def test_rejects_unnormalized_zero_negative_and_nonfinite_probabilities():
    y, p, pairs = _valid_audit_inputs()

    p_unnorm = p.copy()
    p_unnorm[0, 0] = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        forecast_diagnostics(p_unnorm, y[:, 1:])
    with pytest.raises(ValueError):
        audit_sequence_contexts(y, None, p_unnorm, pairs, fit_through=12)

    p_zero = p.copy()
    p_zero[0, 0] = [0.0, 0.5, 0.5]
    with pytest.raises(ValueError):
        forecast_diagnostics(p_zero, y[:, 1:])

    p_nan = p.copy()
    p_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        forecast_diagnostics(p_nan, y[:, 1:])

    p_inf = p.copy()
    p_inf[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        audit_sequence_contexts(y, None, p_inf, pairs, fit_through=12)


def test_rejects_misaligned_simulation_shapes():
    y, p, _ = _valid_audit_inputs(n=4, t=20, k=3, cut=12)
    wrong_shape = np.zeros((4, 7), dtype=int)  # expected suffix length 8
    good_shape = np.zeros((4, 8), dtype=int)
    with pytest.raises(ValueError):
        audit_sequence_contexts(y, None, p, [(wrong_shape, good_shape)], fit_through=12)
    with pytest.raises(ValueError):
        audit_sequence_contexts(y, None, p, [(good_shape, wrong_shape)], fit_through=12)
    with pytest.raises(ValueError):
        audit_sequence_contexts(y, None, p, [], fit_through=12)


def test_rejects_duplicate_and_nonconsecutive_csv_trials(tmp_path):
    dup = tmp_path / "dup.csv"
    dup.write_text("subject,trial,choice\ns1,1,A\ns1,1,B\ns1,2,A\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_trials(dup, ["A", "B"])
    with pytest.raises(ValueError):
        read_ragged_trials(dup, ["A", "B"])

    gap = tmp_path / "gap.csv"
    gap.write_text("subject,trial,choice\ns1,1,A\ns1,3,B\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_trials(gap, ["A", "B"])
    # Ragged reader accepts gaps by marking trial 2 as missing (mask=False)
    choices, mask = read_ragged_trials(gap, ["A", "B"])
    assert choices.shape == (1, 3)
    assert mask.tolist() == [[True, False, True]]


def test_causality_check_catches_future_leakage():
    rng = np.random.default_rng(7)
    y = rng.integers(3, size=(4, 20))

    def leaky_forecast(history):
        # Leaks the next response directly into the probability vector
        n, t = history.shape
        out = np.full((n, t - 1, 3), 0.1)
        out[np.arange(n)[:, None], np.arange(t - 1)[None, :], history[:, 1:]] = 0.8
        return out

    with pytest.raises(ValueError, match="depends on responses"):
        check_forecast_causality(leaky_forecast, y, fit_through=12, categories=3)
