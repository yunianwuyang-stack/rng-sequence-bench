import numpy as np
from scipy.integrate import quad
from scipy.stats import invwishart

from rng_sequence_bench.choice_models import GRID, RDC, design_features
from rng_sequence_bench.separated_prior import SeparatedPopulation, covariance_correction, draw_prior


def test_covariance_ratio_includes_scale_correlation_jacobian():
    scale = np.array([.35, .55, .7])
    d = len(scale)
    reference = np.array([[1., .18, .02], [.18, 1., -.35], [.02, -.35, 1.]])
    a = reference * scale[:, None] * scale
    b = a * 1.5
    for dmatrix in (a, b):
        assert np.linalg.eigvalsh(dmatrix).min() > 0
    scatter = d * np.eye(d)

    def target(cov):
        sd = np.sqrt(np.diag(cov))
        correlation = cov / sd[:, None] / sd
        likelihood = -(12/2) * np.linalg.slogdet(cov)[1] - .5 * np.trace(np.linalg.solve(cov, scatter))
        jacobian = d * np.log(sd).sum()
        return likelihood - .5 * np.sum((sd / scale) ** 2) + np.linalg.slogdet(correlation)[1] - jacobian

    ratio = (target(b) - target(a)) - (invwishart.logpdf(b, df=12, scale=scatter)
                                     - invwishart.logpdf(a, df=12, scale=scatter))
    np.testing.assert_allclose(ratio, covariance_correction(b, scale, 2) - covariance_correction(a, scale, 2))


def test_time_varying_repeat_feature_is_aligned_and_history_causal():
    choices = np.array([[1, 2, 3, 1, 1], [0, 5, 0, 5, 0]])
    feature = design_features(choices, GRID, (*RDC, "repeat_time"))
    target = feature["repeat"] * ((np.arange(1, choices.shape[1]) - 70) / 100)[None, :, None]
    np.testing.assert_allclose(feature["repeat_time"], target)
    modified = choices.copy()
    modified[:, 3:] = 4
    second = design_features(modified, GRID, (*RDC, "repeat_time"))
    np.testing.assert_array_equal(feature["repeat_time"][:, :3], second["repeat_time"][:, :3])


def test_prior_draw_shapes_and_standardized_covariance():
    mean, cov, beta = draw_prior(np.array([.5,.7]), np.array([.3,.9]), 2.0, 30_000,
                                 np.random.default_rng(32))
    np.testing.assert_allclose(beta.mean(axis=0), mean, atol=.01)
    np.testing.assert_allclose(np.cov(beta.T), cov, rtol=.07, atol=.002)


def test_small_population_scale_slice_matches_quadrature():
    population=SeparatedPopulation(np.array([.4]),np.array([.35]),2.0)
    n, ss = 24, .8
    rng=np.random.default_rng(189)
    def unnormalized(sd):
        return np.exp(-n*np.log(sd)-ss/(2*sd*sd)-0.5*(sd/.35)**2)
    expected=quad(lambda sd:sd*unnormalized(sd),.025,1.8)[0]/quad(unnormalized,.025,1.8)[0]
    draws = []
    for _ in range(4200):
        population.cov=population._small_group_covariance(np.array([[ss]]),n,rng)
        draws.append(np.sqrt(population.cov[0,0]))
    np.testing.assert_allclose(np.mean(draws[200:]),expected,atol=.003)
