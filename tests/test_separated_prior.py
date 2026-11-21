import numpy as np
import pytest
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


@pytest.mark.parametrize("prior_jumps",[False,True])
def test_noncentered_scale_update_matches_exact_conditional_mean(prior_jumps):
    prior = .45
    mean = np.zeros(1)
    population = SeparatedPopulation(np.ones(1),np.array([prior]),2.0,prior_jumps=prior_jumps)
    cov = population.cov
    z = .9
    beta = np.array([[z*prior]])
    rng=np.random.default_rng(818821)

    def likelihood(values):
        return -.5*((values[:,0]-.4)/.35)**2

    def density(sd):
        return np.exp(likelihood(np.array([[sd*z]]))[0]-.5*(sd/prior)**2)

    probability = quad(density,0,3)[0]
    expected = quad(lambda sd:sd*density(sd),0,3)[0]/probability
    ll = likelihood(beta)
    values=[]
    for _ in range(10_000):
        cov,ll=population.interweave(likelihood,beta,ll,mean,cov,rng)
        values.append(np.sqrt(cov[0,0]))
    np.testing.assert_allclose(beta[0,0]/values[-1],z)
    np.testing.assert_allclose(ll,likelihood(beta))
    np.testing.assert_allclose(np.mean(values[1000:]),expected,atol=.018)


@pytest.mark.parametrize("prior_jumps",[False,True])
def test_standardized_loading_update_preserves_lkj_marginal(prior_jumps):
    population=SeparatedPopulation(np.ones(2),np.ones(2),2.0,prior_jumps=prior_jumps)
    rng=np.random.default_rng(805)
    cov=population.cov
    beta=np.array([[.3,.7],[-.9,.2]])
    mean=np.zeros(2)
    ll=np.zeros(2)
    correlations=[]
    for _ in range(25_000):
        cov,ll=population.interweave(lambda values:ll,beta,ll,mean,cov,rng)
        correlations.append(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]))
    assert (population.interweave_accepted>0).all()
    np.testing.assert_allclose(np.mean(correlations[1000:]),0,atol=.035)
    np.testing.assert_allclose(np.var(correlations[1000:]),.2,atol=.025)
