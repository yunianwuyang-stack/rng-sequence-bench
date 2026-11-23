"""Load long CSVs of finite-category choices (equal-length or variable-length)."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def from_ragged(sequences: Sequence[Sequence[int]], *, pad_value: int = -1):
    """Pad variable-length integer sequences into (choices, mask) arrays.

    Returns ``(choices, mask)`` of shape ``(N, max_T)``, where ``mask`` is a
    boolean array marking observed trials (``True``) versus right-padding
    (``False``). Missing entries inside a sequence may also be coded with
    ``pad_value`` (default ``-1``) and are marked ``False`` in ``mask``.
    """
    if not sequences:
        raise ValueError("Provide at least one sequence")
    rows = [np.asarray(seq, dtype=int) for seq in sequences]
    if any(r.ndim != 1 or len(r) == 0 for r in rows):
        raise ValueError("Each sequence must be a nonempty 1-D integer array")
    max_len = max(len(r) for r in rows)
    choices = np.full((len(rows), max_len), pad_value, dtype=int)
    mask = np.zeros((len(rows), max_len), dtype=bool)
    for i, r in enumerate(rows):
        choices[i, : len(r)] = r
        mask[i, : len(r)] = r != pad_value
    if not mask.any(axis=1).all():
        raise ValueError("Each participant must have at least one valid choice")
    return choices, mask


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


def read_ragged_trials(path, alphabet, *, missing_tokens=("", "NA", "NaN", "null", "-1")):
    """Load variable-length or missing-trial CSV into ``(choices, mask)``.

    Subjects follow first appearance. Within each subject, trial indices must be
    unique positive integers. Unobserved trial positions up to each subject's
    maximum trial, or rows whose choice matches ``missing_tokens``, receive
    ``-1`` in ``choices`` and ``False`` in ``mask``.
    """
    if not all(alphabet) or len(set(alphabet)) != len(alphabet):
        raise ValueError("Alphabet labels must be nonempty and unique")
    outcomes = {label: i for i, label in enumerate(alphabet)}
    missing = set(missing_tokens)
    people: dict[str, dict[int, int]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"subject", "trial", "choice"} <= set(reader.fieldnames):
            raise ValueError("CSV requires subject, trial and choice columns")
        for row in reader:
            subject, choice = row["subject"], row["choice"]
            if not subject:
                raise ValueError(f"Missing subject on line {reader.line_num}")
            trial = int(row["trial"])
            if trial < 1:
                raise ValueError(f"Trial index must be >= 1 on line {reader.line_num}")
            subject_rows = people.setdefault(subject, {})
            if trial in subject_rows:
                raise ValueError(f"Duplicate trial {trial} for subject {subject!r}")
            if choice in missing:
                subject_rows[trial] = -1
            elif choice in outcomes:
                subject_rows[trial] = outcomes[choice]
            else:
                raise ValueError(f"Unknown choice {choice!r} on line {reader.line_num}")
    if not people:
        raise ValueError("Input must contain at least one subject")
    max_trial = max(max(rows) for rows in people.values())
    choices = np.full((len(people), max_trial), -1, dtype=int)
    mask = np.zeros((len(people), max_trial), dtype=bool)
    for i, rows in enumerate(people.values()):
        for t, val in rows.items():
            if val >= 0:
                choices[i, t - 1] = val
                mask[i, t - 1] = True
    if not mask.any(axis=1).all():
        raise ValueError("Each participant must have at least one valid choice")
    return choices, mask
