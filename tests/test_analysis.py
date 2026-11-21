import json
import subprocess
import sys

import numpy as np
import pytest

from rng_sequence_bench import (
    benchmark,
    fixed_count_test,
    pooling_benchmark,
    transfer_benchmark,
)
from rng_sequence_bench.analysis import _feature_array, _geometry, _scores


def test_fixed_count_expectation():
    data = [[0, 0, 1, 1], [0, 1, 0, 1]]
    summary = fixed_count_test(data, 2, permutations=9, seed=8)
    assert summary["expected_repeat"] == pytest.approx(1 / 3)
    assert summary["observed_repeat"] == pytest.approx(1 / 3)
    assert summary["permutation_p"] == 1
    assert summary["iid_variation_mean"] == pytest.approx(3 / 16)


def test_frequency_features_exclude_current_trial_and_drop_old_choices():
    data = np.array([[0, 0, 1, 2, 1], [0, 0, 1, 0, 0]])
    distances, direction = _geometry([[0], [1], [2]], 3)
    features = _feature_array(data, distances, direction, 4, frequency_window=2)
    np.testing.assert_allclose(features[0, 2, :, 5], [-1, 0, 1] / np.sqrt(3))
    np.testing.assert_allclose(
        features[0, 2, :, 6], [-1 / 3, -1 / 3, 2 / 3] / np.sqrt(2)
    )
    np.testing.assert_allclose(features[0, :3], features[1, :3])


def test_holdout_pooling_scores_each_participant_once():
    data = np.random.default_rng(8).integers(0, 3, size=(7, 12))
    summary = pooling_benchmark(data, [[0], [1], [2]], [5, 8], 3, folds=3)
    assert sorted(summary["folds"]) == [2, 2, 3]
    for model in summary["calibration"].values():
        assert sum(row["n"] for row in model["bins"]) == 7 * 2 * 3


def test_feature_array_does_not_use_current_or_future_responses():
    data = np.array([[0, 1, 2, 1, 2], [2, 0, 1, 0, 0]])
    geometry = _geometry([[0], [1], [2]], 3)
    altered = data.copy()
    altered[:, 3:] = 2
    first = _feature_array(data, *geometry, 4)
    other = _feature_array(altered, *geometry, 4)
    np.testing.assert_array_equal(first[:, :3], other[:, :3])
    assert first[0, 1, 1, 2] == 0
    assert first[0, 1, 2, 4] == 1


def test_uniform_probability_scoring():
    result, bins = _scores(np.full((1, 2, 3), 1 / 3), np.array([[0, 2]]), 10)
    np.testing.assert_allclose(result, [[np.log(3), 2 / 3, 1 / 3]])
    assert bins[:, 0].sum() == 2


def test_transfer_uses_matched_sessions():
    first = np.array([[0, 1, 2, 1, 0, 2], [2, 1, 0, 1, 2, 0]])
    second = np.flip(first, axis=1)
    result = transfer_benchmark(first, second, [[0], [1], [2]], 2, bootstrap=5)
    assert result["training_trials"] == 6
    assert result["score_start"] == 3
    assert result["score_end"] == 6
    assert sum(row["n"] for row in result["calibration"]["RDC"]["bins"]) == 8
    with pytest.raises(ValueError, match="aligned"):
        transfer_benchmark(first, np.vstack((first, first)), [[0], [1], [2]], 2)


def test_cli_and_reliability_counts(tmp_path):
    records = tmp_path / "trials.csv"
    records.write_text(
        "subject,trial,choice\n"
        + "".join(f"{i},{t + 1},{(i + t) % 3}\n" for i in range(5) for t in range(12)),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rng_sequence_bench",
            str(records),
            str(output),
            "--alphabet",
            "0,1,2",
            "--origins",
            "6,9",
            "--width",
            "3",
            "--permutations",
            "5",
            "--bootstrap",
            "5",
            "--simulations",
            "2",
        ],
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    prediction = report["prediction"]
    assert prediction["rolling"][0]["test_start"] == 7
    assert prediction["rolling"][1]["test_end"] == 12
    assert sum(row["n"] for row in prediction["calibration"]["RDC"]["bins"]) == 30
    assert "RDC" in prediction["plug_in_simulation"]
    with pytest.raises(ValueError, match="disjoint"):
        benchmark(np.tile(np.arange(12) % 3, (2, 1)), [[0], [1], [2]], [5, 6], 2)
