import numpy as np

from rps_missingness_sensitivity_r22 import randomize_observed_positions, randomized_position_reference


def test_random_position_reference_preserves_each_persons_observed_values_and_order():
    y = np.array([[0, 1, 0, 0, 2, 0], [2, 1, 0, 0, 0, 0]], dtype=np.int8)
    m = np.array([[1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=bool)
    yr, mr = randomize_observed_positions(y, m, np.random.default_rng(7))
    assert mr.sum(axis=1).tolist() == m.sum(axis=1).tolist()
    for i in range(len(y)):
        assert np.array_equal(yr[i, mr[i]], y[i, m[i]])


def test_random_position_reference_is_reproducible():
    y = np.array([[0, 1, 0, 0, 2, 0]], dtype=np.int8)
    m = np.array([[1, 1, 0, 0, 1, 0]], dtype=bool)
    a = randomized_position_reference(y, m, seed=19, reps=5)
    b = randomized_position_reference(y, m, seed=19, reps=5)
    assert a == b
