"""Load a long CSV of consecutive finite-category choices."""

import csv
from pathlib import Path

import numpy as np


def read_trials(path, alphabet):
    """Return N×T choices in first-appearance subject order and given category order."""
    if not all(alphabet) or len(set(alphabet)) != len(alphabet):
        raise ValueError("Alphabet labels must be nonempty and unique")
    outcomes = {label: i for i, label in enumerate(alphabet)}
    people = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"subject", "trial", "choice"} <= set(reader.fieldnames):
            raise ValueError("CSV requires subject, trial and choice columns")
        for row in reader:
            subject, choice = row["subject"], row["choice"]
            if not subject or choice not in outcomes:
                raise ValueError(f"Missing subject or unknown choice on line {reader.line_num}")
            people.setdefault(subject, []).append((int(row["trial"]), outcomes[choice]))
    sequences = []
    for subject, rows in people.items():
        rows.sort()
        positions, sequence = zip(*rows, strict=True)
        if positions != tuple(range(1, len(rows) + 1)):
            raise ValueError(f"Trial positions for {subject!r} must start at 1 and be consecutive")
        sequences.append(sequence)
    if not sequences or len({len(s) for s in sequences}) != 1:
        raise ValueError("Input must contain at least one subject and equal sequence lengths")
    return np.asarray(sequences, dtype=int)
