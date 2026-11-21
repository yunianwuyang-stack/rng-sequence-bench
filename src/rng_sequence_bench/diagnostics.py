"""Run the posterior package's chain summaries on a standardized CSV."""

import argparse
import shutil
import subprocess
from importlib.resources import as_file, files
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "draws", type=Path, help="CSV of retained draws, identified by chain and draw"
    )
    parser.add_argument("output", type=Path, help="Directory for diagnostics CSV files")
    parser.add_argument("--max-treedepth", type=int, required=True)
    args = parser.parse_args()
    if args.max_treedepth < 1:
        parser.error("max-treedepth must be positive")
    rscript = shutil.which("Rscript")
    if rscript is None:
        parser.error("Rscript and the R package posterior are required")
    with as_file(files("rng_sequence_bench").joinpath("diagnose.R")) as script:
        subprocess.run(
            [
                rscript,
                "--vanilla",
                str(script),
                str(args.draws.resolve(strict=True)),
                str(args.output.resolve()),
                str(args.max_treedepth),
            ],
            check=True,
        )
