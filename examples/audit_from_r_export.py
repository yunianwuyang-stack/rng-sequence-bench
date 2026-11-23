"""Load R/Stan-exported CSVs and run audit_sequence_contexts."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from rng_sequence_bench import audit_sequence_contexts


def read_int_matrix(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return np.asarray([[int(x) for x in r] for r in rows[1:]], dtype=int)


def main(folder: Path, fit_through: int = 25):
    choices = read_int_matrix(folder / "choices.csv")
    n, t_len = choices.shape
    with (folder / "forecasts.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        p_cols = [c for c in reader.fieldnames if c.startswith("p")]
        k = len(p_cols)
        probs = np.zeros((n, t_len - 1, k), dtype=float)
        for row in reader:
            i = int(row["subject"]) - 1
            t = int(row["trial"]) - 2
            probs[i, t] = [float(row[c]) for c in p_cols]
    reset_files = sorted(folder.glob("reset_*.csv"))
    prefix_files = sorted(folder.glob("prefix_*.csv"))
    pairs = [
        (read_int_matrix(rf), read_int_matrix(pf))
        for rf, pf in zip(reset_files, prefix_files, strict=True)
    ]
    report = audit_sequence_contexts(
        choices, None, probs, pairs, fit_through=fit_through, initial_transitions=4
    )
    print(json.dumps(report["holdout_scores_named"], indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
