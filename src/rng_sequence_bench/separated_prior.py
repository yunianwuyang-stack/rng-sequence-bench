"""Separation prior for Gaussian choice coefficients and an exact MH update."""

import numpy as np
from scipy.stats import invwishart, wishart

from .hierarchy import _slice_normalized


def draw_prior(mean_scale, sd_scale, eta, people, rng, mean_location=0.0):
    """Generate population and participant effects under Normal/HalfNormal/LKJ."""
    dimension = len(sd_scale)
    mean = rng.normal(loc=mean_location,scale=mean_scale)
    sd = np.abs(rng.normal(scale=sd_scale))
    if dimension == 1:
        corr = np.ones((1, 1))
    else:
        gram = wishart.rvs(df=dimension - 1 + 2 * eta, scale=np.eye(dimension),
                            random_state=rng)
        gram_sd = np.sqrt(np.diag(gram))
        corr = gram / gram_sd[:, None] / gram_sd
    cov = corr * sd[:, None] * sd
    beta = mean + rng.normal(size=(people, dimension)) @ (sd[:, None] * np.linalg.cholesky(corr)).T
    return mean, cov, beta


def covariance_correction(cov, sd_scale, eta):
    """Log ratio of the separation prior to the Jeffreys working prior."""
    dimension = len(sd_scale)
    sd = np.sqrt(np.diag(cov))
    log_sd = np.log(sd).sum()
    log_det = np.linalg.slogdet(cov)[1]
    log_corr = log_det - 2 * log_sd
    return (eta - 1 + (dimension + 1) / 2) * log_corr + log_sd - 0.5 * np.sum((sd / sd_scale) ** 2)


class SeparatedPopulation:
    """Blocked update of independent mean and log-scale/correlation prior."""

    def __init__(self, mean_sd, between_sd, eta, mean_location=0.0, prior_jumps=False):
        self.mean_precision = np.diag(1 / mean_sd ** 2)
        self.mean_natural = self.mean_precision @ np.broadcast_to(mean_location,mean_sd.shape)
        self.between_sd = between_sd
        self.eta = eta
        self.prior_jumps = prior_jumps
        self.cov = np.diag(between_sd ** 2)
        self.accepted = self.attempts = 0
        self.interweave_accepted = np.zeros(2,dtype=int)
        self.interweave_attempts = np.zeros(2,dtype=int)

    def update(self, beta, rng):
        n, dimension = beta.shape
        total = beta.sum(axis=0)
        likelihood_precision = np.linalg.inv(self.cov)
        precision = n * likelihood_precision + self.mean_precision
        lower = np.linalg.cholesky(precision)
        mean = np.linalg.solve(precision, likelihood_precision @ total + self.mean_natural)
        mean += np.linalg.solve(lower.T, rng.normal(size=dimension))

        residuals = beta - mean
        scatter = residuals.T @ residuals
        if n < 100:
            self.cov = self._small_group_covariance(scatter, n, rng)
            return mean, self.cov
        candidate = np.asarray(invwishart.rvs(df=n,scale=scatter,random_state=rng)).reshape(dimension,dimension)
        log_ratio = (covariance_correction(candidate,self.between_sd,self.eta)
                     - covariance_correction(self.cov,self.between_sd,self.eta))
        self.attempts += 1
        if np.log(rng.random()) < log_ratio:
            self.cov = candidate
            self.accepted += 1
        return mean, self.cov

    def interweave(self, likelihood, beta, ll, mean, cov, rng):
        """Update either one scale or a loading in standardized coordinates."""
        dimension = len(mean)
        j = int(rng.integers(dimension))
        sd = np.sqrt(np.diag(cov))
        corr = cov / sd[:,None] / sd
        new_sd = sd.copy()
        proposed = beta.copy()
        scale_update = dimension == 1 or rng.random() < .75
        if scale_update:
            if self.prior_jumps and rng.random() < .45:
                new_sd[j] = abs(rng.normal(scale=self.between_sd[j]))
                multiplier = new_sd[j]/sd[j]
                log_prior = 0.
            else:
                log_step = rng.normal(scale=.38)
                multiplier = np.exp(log_step)
                new_sd[j] *= multiplier
                log_prior = log_step - .5 * (new_sd[j]**2-sd[j]**2) / self.between_sd[j]**2
            proposed[:,j] = mean[j] + (beta[:,j]-mean[j]) * multiplier
        else:
            rest = np.arange(dimension) != j
            lower = np.linalg.cholesky(corr[np.ix_(rest,rest)])
            row = np.linalg.solve(lower,corr[rest,j])
            change = row.copy()
            k = int(rng.integers(len(change)))
            prior_jump = self.prior_jumps and rng.random() < .30
            if prior_jump:
                available = 1-row @ row+row[k]**2
                change[k] = np.sqrt(available)*(2*rng.beta(self.eta,self.eta)-1)
            else:
                change[k] += rng.normal(scale=.16)
            current_variance = 1-row @ row
            next_variance = 1-change @ change
            self.interweave_attempts[1] += 1
            if next_variance <= 0:
                return cov,ll
            others = np.linalg.solve(lower,((beta[:,rest]-mean[rest])/sd[rest]).T).T
            noise = ((beta[:,j]-mean[j])/sd[j]-others @ row)/np.sqrt(current_variance)
            proposed[:,j] = mean[j] + sd[j] * (others @ change + np.sqrt(next_variance)*noise)
            log_prior = 0. if prior_jump else (self.eta-1) * np.log(next_variance/current_variance)
            corr[rest,j] = corr[j,rest] = lower @ change
        if scale_update:
            self.interweave_attempts[0] += 1
        candidate_ll = likelihood(proposed)
        if np.log(rng.random()) < log_prior + np.sum(candidate_ll-ll):
            self.interweave_accepted[0 if scale_update else 1] += 1
            beta[:,j] = proposed[:,j]
            self.cov = corr * new_sd[:,None] * new_sd
            return self.cov,candidate_ll
        return cov,ll

    def _small_group_covariance(self, scatter, n, rng):
        """Slice log scales and compose symmetric correlation updates."""
        d = len(self.between_sd)
        sd = np.sqrt(np.diag(self.cov))
        corr = self.cov / sd[:,None] / sd
        quadratic = np.linalg.inv(corr) * scatter
        inverse_sd = 1.0 / sd
        for j in rng.permutation(d):
            cross = quadratic[j] @ inverse_sd - quadratic[j,j] * inverse_sd[j]

            def density(log_sd):
                scaled = np.exp(-log_sd)
                likelihood = quadratic[j,j] * scaled**2 + 2 * cross * scaled
                prior = (np.exp(log_sd)/self.between_sd[j])**2
                return -(n-1)*log_sd - 0.5*(likelihood+prior)

            point = _slice_normalized(density,np.log(sd[j]),0.28,rng)
            sd[j] = np.exp(point)
            inverse_sd[j] = 1.0 / sd[j]
        normalized = scatter * inverse_sd[:,None] * inverse_sd
        if d == 1:
            return np.diag(sd**2)

        def correlation_density(matrix):
            try:
                factor = np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                return -np.inf
            log_det = 2 * np.log(np.diag(factor)).sum()
            trace = np.trace(np.linalg.solve(matrix, normalized))
            return (self.eta-1-n/2)*log_det - 0.5*trace

        coordinate = np.column_stack(np.tril_indices(d,-1))
        current = correlation_density(corr)
        for position in rng.permutation(len(coordinate))[:min(14,len(coordinate))]:
            i,j=coordinate[position]
            candidate=corr.copy()
            candidate[i,j]=candidate[j,i]=corr[i,j]+rng.normal(scale=.14)
            proposed=correlation_density(candidate)
            self.attempts += 1
            if np.log(rng.random()) < proposed-current:
                corr,current=candidate,proposed
                self.accepted += 1
        return corr*sd[:,None]*sd
