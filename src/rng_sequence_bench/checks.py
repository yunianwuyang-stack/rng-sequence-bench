"""Diagnose fixed forecasts and/or sequences stored as NumPy arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import forecast_diagnostics
from .sequence_metrics import sequence_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path, help="NPZ file with the documented arrays")
    parser.add_argument("output", type=Path, help="JSON output path")
    args = parser.parse_args()
    report = {}
    with np.load(args.arrays, allow_pickle=False) as data:
        if "probabilities" in data or "targets" in data:
            if not {"probabilities", "targets"} <= set(data.files):
                parser.error("Forecast checks require both probabilities and targets")
            report["forecasts"] = forecast_diagnostics(
                data["probabilities"], data["targets"]
            )
        if "choices" in data or "coordinates" in data:
            if not {"choices", "coordinates"} <= set(data.files):
                parser.error("Sequence checks require both choices and coordinates")
            reference = data.get("reference", None)
            report["sequences"] = sequence_diagnostics(
                data["choices"], data["coordinates"], reference
            )
    if not report:
        parser.error("No forecast or sequence arrays were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
