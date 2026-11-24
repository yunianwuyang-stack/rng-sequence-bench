import numpy as np
import pytest
from rng_sequence_bench import check_forecast_causality, check_prefix_preprocessing, mc_ranking_probability


def test_all_positions_detect_hidden_leak():
    y = np.zeros((2, 8), dtype=int)
    def leak(x):
        p = np.full((2, 7, 2), .5)
        p[:, 4, 0] += (x[:, 5] - x[:, 6]) * .1
        p[:, 4, 1] = 1 - p[:, 4, 0]
        return p
    with pytest.raises(ValueError, match="depends"):
        check_forecast_causality(leak, y, 3, 2, mode="all")
    fixed = lambda x: np.full((2, 7, 2), .5)
    report = check_forecast_causality(fixed,y,3,2,mode="random",n_positions=3,seed=7)
    assert len(set(report["altered_trials"])) == 3
    assert report == check_forecast_causality(fixed,y,3,2,mode="random",n_positions=3,seed=7)


def test_preprocessing_sees_only_prefix():
    y = np.array([[0, 0, 1, 1]])
    assert check_prefix_preprocessing(lambda x,c: {"scale": x[:, :c].mean()}, y, 2)["checked_fields"] == ["scale"]
    with pytest.raises(ValueError, match="scale"):
        check_prefix_preprocessing(lambda x,c: {"scale": x.mean()}, y, 2)


def test_distinct_mc_estimands_and_ties():
    result = mc_ranking_probability([0, 0, 0, 1], [1,1,1,0])
    assert result["empirical_win_rate"] == .75
    assert result["normal_sign_support_for_mean"] != .75
    tied = mc_ranking_probability([1,1], [1,1])
    assert (tied["empirical_win_rate"], tied["tie_rate"], tied["normal_sign_support_for_mean"]) == (0,1,.5)
