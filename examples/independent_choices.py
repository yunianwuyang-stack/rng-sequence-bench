"""A runnable 3-category IID control for the API shown in the article."""

import numpy as np
from rng_sequence_bench import audit_sequence_contexts

rng = np.random.default_rng(21)
n, trials, k, cut = 8, 30, 3, 20
choices = rng.integers(k, size=(n, trials))
coordinates = np.arange(k)[:, None]
probabilities = np.full((n, trials - 1, k), 1 / k)
simulated = rng.integers(k, size=(16, n, trials - cut))
pairs = ((suffix, suffix.copy()) for suffix in simulated)
report = audit_sequence_contexts(
    choices, coordinates, probabilities, pairs,
    fit_through=cut, initial_transitions=3,
)
print(report["holdout_forecasts"]["scores"])
