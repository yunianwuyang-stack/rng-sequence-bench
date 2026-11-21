"""A runnable 3-category IID control for the API shown in the article."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from rng_sequence_bench import audit_sequence_contexts, read_trials

rng = np.random.default_rng(21)
n, trials, k, cut = 8, 30, 3, 20
labels = ("A", "B", "C")
with TemporaryDirectory() as folder:
    path = Path(folder) / "trials.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("subject", "trial", "choice"))
        for i, row in enumerate(rng.integers(k, size=(n, trials))):
            writer.writerows((i, t, labels[c]) for t, c in enumerate(row, 1))
    choices = read_trials(path, labels)
coordinates = np.arange(k)[:, None]
probabilities = np.full((n, trials - 1, k), 1 / k)
simulated = rng.integers(k, size=(16, n, trials - cut))
pairs = ((suffix, suffix.copy()) for suffix in simulated)
report = audit_sequence_contexts(
    choices, coordinates, probabilities, pairs,
    fit_through=cut, initial_transitions=3,
)
expected = [np.log(3), 2 / 3, 1 / 3]
assert np.allclose(report["holdout_forecasts"]["scores"], expected)
print(report["holdout_forecasts"]["scores"])
