import numpy as np
import pytest

from rng_sequence_bench.choice_models import (
    GRID,
    RDC,
    design,
    design_features,
    simulate,
)
from rng_sequence_bench.hierarchy import (
    Likelihood,
    _repeat_slice_update,
    run_hierarchy_chain,
)
from rng_sequence_bench.predictive import (
    posterior_predictive_checks,
    recurrence_distribution,
)
from rng_sequence_bench.recovery import one_replicate


def test_prefix_and_zero_effects():
    rng = np.random.default_rng(112)
    u = rng.random((8, 22))
    coefficients = np.zeros((8, 3))
    free = simulate(coefficients, RDC, GRID, u)
    np.testing.assert_array_equal(free, (u * 9).astype(int))
    observed = np.ones((len(u), 12), dtype=int)
    frozen = simulate(coefficients, RDC, GRID, u, prefix=observed)
    np.testing.assert_array_equal(frozen[:, :12], observed)
    np.testing.assert_array_equal(frozen[:, 12:], free[:, 12:])


def test_gaps_are_within_participant_and_category():
    actual = np.array([[0, 1, 0, 0], [1, 0, 1, 0]])
    assert recurrence_distribution(actual, 2) == [0.25, 0.75, 0, 0, 0, 0, 0]
    checks = posterior_predictive_checks(actual, [actual, actual], [[0], [1]])
    assert checks["distance_from_observed"]["transition_js"]["median"] == 0
    assert checks["same_person_coverage95"] == {"tv": 1.0, "repeat": 1.0}


def test_hierarchical_chain_emits_every_retained_draw():
    rng = np.random.default_rng(113)
    beta = rng.normal(size=(12, 3)) * 0.35
    data = simulate(beta, RDC, GRID, rng.random((len(beta), 32)))
    x = design(design_features(data, GRID, RDC), RDC)
    loglik = Likelihood(x, data[:, 1:])
    np.testing.assert_allclose(loglik(np.zeros_like(beta)), -31 * np.log(9))
    recorded = []

    def record(i, mean, cov, subject, ll):
        assert np.all(np.linalg.eigvalsh(cov) > 0)
        assert mean.shape == (3,)
        np.testing.assert_allclose(loglik(subject), ll)
        recorded.append(i)

    rate = run_hierarchy_chain(
        x,
        data[:, 1:],
        prior_sd=[1.0, 0.55, 0.60],
        warmup=20,
        draws=40,
        seed=4,
        record=record,
    )
    assert recorded == list(range(40))
    assert np.all((rate >= 0) & (rate <= 1))


@pytest.mark.parametrize("bad_choice", [-1, 3, 2**32, 1.5])
def test_likelihood_rejects_bad_input_before_loading_native_code(bad_choice):
    features = np.zeros((2, 2, 3, 1))
    choices = np.full((2, 2), bad_choice)
    with pytest.raises(ValueError):
        Likelihood(features, choices, native="unavailable.so")
    valid = Likelihood(features, np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="participant by coefficient"):
        valid(np.zeros((1, 2)))


def test_recovery_records_truth_and_winners():
    row = one_replicate(("repeat", 6, 14, 1.35, 1, 19, False))
    assert row["parameters"]["names"] == ("repeat",)
    assert row["forecast_winner"] in row["candidates"]
    assert row["candidates"]["iid"][0] == pytest.approx(np.log(9))


def test_repeat_slice_agrees_with_exact_conditional_quadrature():
    # With no repeats, the posterior remains proper under the Gaussian prior.
    x = np.array([[[[1.0, 0.0], [0.0, v]] for v in [0.2, 1.1, -0.4, 0.8]]])
    likelihood = Likelihood(x, np.ones((1, 4), dtype=int))
    rng = np.random.default_rng(900)
    precision = np.array([[1.5, -0.2], [-0.2, 0.9]])
    mean = np.array([-0.3, 0.4])
    beta = np.array([[0.0, 0.6]])
    grid = np.linspace(-10, 10, 2001)
    adjustments = (
        np.column_stack((grid, np.full_like(grid, beta[0, 1]))) - mean
    )
    lp = np.array([likelihood(np.array([[value, beta[0, 1]]]))[0] for value in grid])
    lp -= 0.5 * np.einsum("ni,ij,nj->n", adjustments, precision, adjustments)
    weights = np.exp(lp - lp.max())
    expected = np.average(grid, weights=weights)
    values = []
    for _ in range(5000):
        beta = _repeat_slice_update(likelihood, beta, mean, precision, rng)
        values.append(beta[0, 0])
    assert beta[0, 1] == 0.6
    assert np.mean(values[200:]) == pytest.approx(expected, abs=0.06)
