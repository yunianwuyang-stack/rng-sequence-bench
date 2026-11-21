"""Command-line entry point for ordered categorical trials."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rng_sequence_bench.analysis import benchmark, fixed_count_test
from rng_sequence_bench.chronological import standardized_benchmark


def read_trials(path, alphabet):
    if not all(alphabet) or len(set(alphabet)) != len(alphabet):
        raise ValueError("Alphabet labels must be nonempty and unique")
    outcomes = {label: i for i, label in enumerate(alphabet)}
    people = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"subject", "trial", "choice"} <= set(
            reader.fieldnames
        ):
            raise ValueError("CSV requires subject, trial and choice columns")
        for row in reader:
            subject, choice = row["subject"], row["choice"]
            if not subject or choice not in outcomes:
                raise ValueError(
                    f"Missing subject or unknown choice on line {reader.line_num}"
                )
            people.setdefault(subject, []).append((int(row["trial"]), outcomes[choice]))
    sequences = []
    for subject, rows in people.items():
        rows.sort()
        positions, sequence = zip(*rows, strict=True)
        if positions != tuple(range(1, len(rows) + 1)):
            raise ValueError(
                f"Trial positions for {subject!r} must start at 1 and be consecutive"
            )
        sequences.append(sequence)
    if not sequences or len({len(s) for s in sequences}) != 1:
        raise ValueError(
            "Input must contain at least one subject and equal sequence lengths"
        )
    return np.asarray(sequences, dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, help="CSV with subject, trial, choice")
    parser.add_argument("output", type=Path, help="JSON output path")
    parser.add_argument(
        "--alphabet",
        required=True,
        help="Comma-separated outcome labels in coordinate order",
    )
    parser.add_argument(
        "--coordinates", type=Path, help="JSON row vectors, one per outcome"
    )
    parser.add_argument(
        "--origins", required=True, help="Comma-separated training prefix lengths"
    )
    parser.add_argument(
        "--width", required=True, type=int, help="Future scoring-window length"
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--half-life", type=float, default=4.0)
    parser.add_argument(
        "--frequency-window",
        type=int,
        help="Add candidate frequency models with this local window",
    )
    parser.add_argument(
        "--include-pooled",
        action="store_true",
        help="Score pooled RD and RDC on the same holdouts",
    )
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument(
        "--standardized",
        action="store_true",
        help="Use inner-tuned ridge and Markov baselines",
    )
    parser.add_argument(
        "--inner-width", type=int, default=25, help="Trials in each inner tuning window"
    )
    args = parser.parse_args()
    alphabet = args.alphabet.split(",")
    coordinates = (
        json.loads(args.coordinates.read_text(encoding="utf-8"))
        if args.coordinates
        else [[i] for i in range(len(alphabet))]
    )
    values = read_trials(args.trials, alphabet)
    origins = [int(i) for i in args.origins.split(",")]
    if args.standardized:
        prediction = standardized_benchmark(
            values,
            coordinates,
            origins,
            [args.width] * len(origins),
            inner_width=args.inner_width,
            seed=args.seed,
        )
    else:
        prediction = benchmark(
            values,
            coordinates,
            origins,
            args.width,
            ridge=args.ridge,
            half_life=args.half_life,
            bootstrap=args.bootstrap,
            simulations=args.simulations,
            seed=args.seed,
            frequency_window=args.frequency_window,
            include_pooled=args.include_pooled,
        )
    summary = {
        "alphabet": alphabet,
        "coordinates": coordinates,
        "coordinate_kind": "supplied" if args.coordinates else "ordinal",
        "order": fixed_count_test(values, len(alphabet), args.permutations, args.seed),
        "prediction": prediction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
