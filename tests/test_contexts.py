import numpy as np
import pytest

from rng_sequence_bench import audit_sequence_contexts, check_forecast_causality


def test_matched_contexts_exclude_boundary_and_keep_run_denominator():
    choices = np.array([[0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0]])
    coords = np.array([[0.0], [1.0]])
    p = np.full((2, 6, 2), 0.5)
    suffix = choices[:, 3:]
    report = audit_sequence_contexts(
        choices, coords, p, [(suffix, suffix)], fit_through=3, initial_transitions=1
    )
    repeat = report["observed_history_repetition"]
    assert repeat["fit"]["observed"] == 0.25
    assert repeat["holdout"]["observed"] == pytest.approx(2 / 6)
    assert report["uniform_minus_prefix_repeat"]["median"] == [0, 0]
    assert report["uniform_start"]["statistics"]["run_survival"]["observed"] == [
        2 / 6,
        0,
        0,
    ]
    with pytest.raises(ValueError, match="suffix shape"):
        audit_sequence_contexts(
            choices, coords, p, [(choices, suffix)], fit_through=3, initial_transitions=1
        )


def test_causality_accepts_lag_features_and_rejects_current_target():
    choices = np.array([[0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0]])

    def forecast(y):
        return 0.2 + 0.6 * np.eye(2)[y[:, :-1]]

    def leaky(y):
        return 0.2 + 0.6 * np.eye(2)[y[:, 1:]]

    assert check_forecast_causality(forecast, choices, 3, 2)["altered_trials"] == [4, 6]
    with pytest.raises(ValueError, match="trial 4"):
        check_forecast_causality(leaky, choices, 3, 2)
