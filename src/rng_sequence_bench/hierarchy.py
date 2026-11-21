"""Exact population Gibbs updates and per-participant independence MH.

The local quadratic expansion controls proposal efficiency; MH acceptance uses
the categorical likelihood and the current population Gaussian density.
"""

from ctypes import CDLL, c_int

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.special import logsumexp
from scipy.stats import invwishart

from .choice_models import fit_batch


class Likelihood:
    """Stable, vectorized likelihood with an optional compiled C++ backend."""

    def __init__(self, x, y, native=None):
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        target = np.asarray(y)
        if self.x.ndim != 4 or min(self.x.shape) < 1 or self.x.shape[2] < 2:
            raise ValueError(
                "Expected nonempty participant by trial by category by coefficient features"
            )
        if not np.isfinite(self.x).all():
            raise ValueError("Features must be finite")
        if target.shape != self.x.shape[:2] or not np.issubdtype(
            target.dtype, np.integer
        ):
            raise ValueError("Choices must be an integer participant by trial matrix")
        if target.min() < 0 or target.max() >= self.x.shape[2]:
            raise ValueError("Choice index is outside the candidate set")
        self.y = np.ascontiguousarray(target, dtype=np.int32)
        self.native = None
        if native is not None:
            function = CDLL(str(native)).subject_log_likelihood
            real = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
            integer = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
            function.argtypes = [real, integer, real, real, c_int, c_int, c_int, c_int]
            function.restype = None
            self.native = function

    def __call__(self, beta):
        beta = np.asarray(beta, dtype=np.float64)
        if beta.shape != (self.x.shape[0], self.x.shape[-1]):
            raise ValueError("Coefficients must be participant by coefficient")
        if self.native is not None:
            out = np.empty(len(beta))
            self.native(self.x, self.y, np.ascontiguousarray(beta), out, *self.x.shape)
            return out
        eta = np.einsum("ntkd,nd->ntk", self.x, beta)
        selected = np.take_along_axis(eta, self.y[..., None], axis=2)[..., 0]
        return np.sum(selected - logsumexp(eta, axis=2), axis=1)


def sample_population(beta, kappa, df, prior_scale, rng):
    """Draw the inverse-Wishart/Gaussian pair conditional on participant effects."""
    n, d = beta.shape
    average = beta.mean(axis=0)
    centered = beta - average
    scale = prior_scale + centered.T @ centered
    scale += (kappa * n / (kappa + n)) * np.outer(average, average)
    cov = np.asarray(invwishart.rvs(df=df + n, scale=scale, random_state=rng)).reshape(
        d, d
    )
    mean = average * n / (n + kappa)
    mean += np.linalg.cholesky(cov / (n + kappa)) @ rng.normal(size=d)
    return mean, cov


def quadratic_proposal(hessian, natural, mean, cov):
    prior_precision = np.linalg.inv(cov)
    precision = hessian + prior_precision
    center = np.linalg.solve(precision, (natural + prior_precision @ mean)[..., None])[
        ..., 0
    ]
    lower = np.linalg.cholesky(precision)
    return precision, center, lower, prior_precision


def log_student(beta, center, precision, df):
    delta = beta - center
    squared = np.einsum("ni,nij,nj->n", delta, precision, delta)
    return -0.5 * (df + beta.shape[1]) * np.log1p(squared / df)


def normal_difference(new, old, mean, precision):
    delta = new - old
    combined = new + old - 2 * mean
    return -0.5 * np.einsum("ni,ij,nj->n", delta, precision, combined)


def _slice_normalized(log_density, current, width, rng):
    """Univariate stepping-out slice update with a randomized initial bracket."""
    level = log_density(current) - rng.exponential()
    left = current - width * rng.random()
    right = left + width
    while log_density(left) > level:
        left -= width
    while log_density(right) > level:
        right += width
    while True:
        proposed = rng.uniform(left, right)
        if log_density(proposed) >= level:
            return proposed
        if proposed < current:
            left = proposed
        else:
            right = proposed


def _repeat_slice_update(likelihood, beta, mean, precision, rng):
    """Update selected repeat effects conditional on all other effects."""
    mask = likelihood.x[..., 0].astype(bool)
    eta = np.einsum("ntkd,nd->ntk", likelihood.x[..., 1:], beta[:, 1:])
    value_repeat = np.sum(np.where(mask, eta, 0.0), axis=-1)
    alternative = logsumexp(np.where(mask, -np.inf, eta), axis=-1)
    gap = value_repeat - alternative
    observed_repeat = np.take_along_axis(mask, likelihood.y[..., None], axis=-1)[..., 0]
    signed = np.where(observed_repeat, -1.0, 1.0)
    p = precision[0, 0]
    center = mean[0] - (beta[:, 1:] - mean[1:]) @ precision[0, 1:] / p
    for i, (intercept, direction, mu) in enumerate(zip(gap, signed, center, strict=True)):
        def log_density(value):
            return -0.5 * p * (value - mu) ** 2 - np.logaddexp(
                0.0, direction * (intercept + value)
            ).sum()

        beta[i, 0] = _slice_normalized(log_density, beta[i, 0], 1.0 / np.sqrt(p), rng)
    return beta


def run_hierarchy_chain(
    x,
    y,
    *,
    prior_sd,
    warmup,
    draws,
    seed,
    native=None,
    record=None,
    report=None,
    mh_updates=1,
    repeat_slice_people=(),
    population_update=None,
):
    """Sample a centered multivariate-normal hierarchy with proper NIW prior.

    The prior is Sigma ~ IW(d+7, 6 diag(prior_sd**2)), and
    mu | Sigma ~ N(0, Sigma/.25). The output callback receives iteration,
    mu, Sigma, the participant effects, and training log likelihood.
    Optionally add a conditional slice update for the repeat coefficient of
    selected participants (zero-based indices). Coefficient zero must be a
    binary indicator for exactly one repeat candidate per trial.
    An optional population update replaces the default conjugate update.
    """
    likelihood = Likelihood(x, y, native=native)
    n, _, _, d = likelihood.x.shape
    prior_sd = np.asarray(prior_sd, dtype=float)
    if (
        prior_sd.shape != (d,)
        or not np.isfinite(prior_sd).all()
        or np.any(prior_sd <= 0)
    ):
        raise ValueError("Require one positive finite prior scale per coefficient")
    if warmup < 0 or draws < 1 or mh_updates < 1:
        raise ValueError("Invalid MCMC iteration count")
    repeats = tuple(repeat_slice_people)
    repeat_likelihood = None
    if repeats:
        if len(set(repeats)) != len(repeats) or any(i < 0 or i >= n for i in repeats):
            raise ValueError("Repeat-slice participant indices must be unique and in range")
        repeat_indices = np.asarray(repeats, dtype=int)
        x_repeat = likelihood.x[repeat_indices]
        repeat_feature = x_repeat[..., 0]
        if not np.isin(repeat_feature, (0.0, 1.0)).all() or not np.all(
            repeat_feature.sum(axis=-1) == 1
        ):
            raise ValueError("Coefficient zero must identify exactly one repeat candidate")
        repeat_likelihood = Likelihood(x_repeat, likelihood.y[repeat_indices], native=native)
    rng = np.random.default_rng(seed)
    kappa, df, tail = 0.25, d + 7, 7.0
    scale = 6 * np.diag(prior_sd**2)
    mode, cov_mode = fit_batch(likelihood.x, likelihood.y, 1.0, covariance=True)
    precision_mode = np.linalg.inv(cov_mode)
    hessian = precision_mode - np.eye(d)
    natural = np.einsum("nij,nj->ni", precision_mode, mode)
    beta = (
        mode
        + np.einsum(
            "nij,nj->ni", np.linalg.cholesky(cov_mode), rng.normal(size=mode.shape)
        )
        * 0.5
    )
    ll = likelihood(beta)
    reference = Likelihood(likelihood.x[:8], likelihood.y[:8])(beta[:8])
    np.testing.assert_allclose(ll[:8], reference, rtol=3e-12, atol=3e-9)
    accept = np.zeros(n, dtype=int)

    for iteration in range(warmup + draws):
        if population_update is None:
            mean, cov = sample_population(beta, kappa, df, scale, rng)
        else:
            mean, cov = population_update(beta, rng)
        precision, center, lower, prior_precision = quadratic_proposal(
            hessian, natural, mean, cov
        )
        for _ in range(mh_updates):
            noise = np.linalg.solve(lower.swapaxes(1, 2), rng.normal(size=(n, d, 1)))[
                ..., 0
            ]
            proposal = (
                center + noise * np.sqrt(tail / rng.chisquare(tail, size=n))[:, None]
            )
            candidate_ll = likelihood(proposal)
            log_ratio = (
                candidate_ll
                - ll
                + normal_difference(proposal, beta, mean, prior_precision)
                + log_student(beta, center, precision, tail)
                - log_student(proposal, center, precision, tail)
            )
            accepted = np.log(rng.random(n)) < log_ratio
            beta[accepted] = proposal[accepted]
            ll[accepted] = candidate_ll[accepted]
            if iteration >= warmup:
                accept += accepted
        if repeat_likelihood is not None:
            values = _repeat_slice_update(
                repeat_likelihood, beta[repeat_indices], mean, prior_precision, rng
            )
            beta[repeat_indices] = values
            ll[repeat_indices] = repeat_likelihood(values)
        if record is not None and iteration >= warmup:
            record(iteration - warmup, mean, cov, beta, ll)
        if report is not None and (iteration + 1) % 100 == 0:
            report(iteration + 1, warmup + draws)

    return accept / (draws * mh_updates)
