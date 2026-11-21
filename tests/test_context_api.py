import numpy as np
import pytest

from rng_sequence_bench import audit_sequence_contexts, rolling_forecasts


def test_audit_without_coordinates_keeps_declared_alphabet():
    choices = np.array([[0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 1, 0]])
    probability = np.full((2, 5, 3), 1/3)
    suffix = choices[:, 2:].copy()
    result = audit_sequence_contexts(choices, probabilities=probability, replicated_pairs=[(suffix, suffix)],
                                     fit_through=2, initial_transitions=1)
    spatial = result["uniform_start"]
    assert len(spatial["statistics"]["transition_joint"]["observed"]) == 3
    assert "distance_distribution" not in spatial["statistics"]
    assert np.isclose(result["holdout_forecasts"]["scores"][0], np.log(3))


def test_rolling_forecasts_sees_each_prefix_once():
    data = np.tile([0, 1], (3, 10))
    lengths = []

    def fit(prefix):
        lengths.append(prefix.shape[1])
        return lambda row: np.full((len(row), row.shape[1]-1, 2), .5)

    result = rolling_forecasts(data, fit, [5, 10, 13], 5, categories=2)
    assert lengths == [5, 10, 13]
    assert result[13]["score_trials"] == [14, 18]
    assert result[5]["scores_ci"]["mean"][0] == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        rolling_forecasts(data, fit, [18], 5, categories=2)
