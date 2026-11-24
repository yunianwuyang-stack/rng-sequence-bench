#!/usr/bin/env python3
"""Compare a conventional single-stat posterior-predictive coverage screen with the
multi-diagnostic descriptive audit using the existing five-scenario calibration grid.

This script re-analyses the precomputed 120-dataset/cell Monte Carlo results. It does
not refit models or create new simulations. Joint-screen rates are reported as
Frechet bounds because the archived aggregate file contains marginal rates, not
per-dataset joint indicator vectors.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def frechet_bounds(rates):
    vals = list(map(float, rates))
    return [max(0.0, sum(vals) - (len(vals) - 1)), min(vals)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "results/adequacy_calibration_validation.json")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results/ppc_comparator_audit_r22.json")
    a = p.parse_args()
    src = json.loads(a.input.read_text(encoding="utf-8"))
    rows = []
    for cell in src["cells"]:
        criteria = {
            "reference_band_tv95": cell["band_pass_rate_tv95"],
            "tv95_practical_margin": cell["predictive_margin_pass_rate_tv95"],
            "tvmean_practical_margin": cell["predictive_margin_pass_rate_tvmean"],
            "transition_js": 1.0 - cell["js_violation_rate"],
        }
        rows.append({
            "scenario": cell["scenario"], "N": cell["N"], "T": cell["T"],
            "R": cell["replicates"], "datasets": cell["datasets"],
            "classic_single_stat_ppc_pass": cell["rep95_coverage_of_obs_tv95"],
            "reference_band_pass_tv95": cell["band_pass_rate_tv95"],
            "tv95_practical_margin_pass": cell["predictive_margin_pass_rate_tv95"],
            "tvmean_practical_margin_pass": cell["predictive_margin_pass_rate_tvmean"],
            "transition_js_pass": 1.0 - cell["js_violation_rate"],
            "joint_descriptive_screen_frechet_lower": frechet_bounds(criteria.values())[0],
            "joint_descriptive_screen_frechet_upper": frechet_bounds(criteria.values())[1],
            "joint_screen_exact_rate_available": False,
        })
    out = {
        "source_file": a.input.name,
        "source_seed": src["seed"],
        "datasets_per_cell": src["datasets_per_cell"],
        "scenarios": sorted({r["scenario"] for r in rows}),
        "definition": {
            "classic_single_stat_ppc": "Observed TV95 lies within the 95% interval of replicated TV95 values.",
            "multi_diagnostic_screen": "Reference-band location, predictive-distribution practical-margin screens for TV95 and TVmean, and transition-JS screen are reported separately.",
            "joint_rate": "Only Frechet bounds are available from marginal aggregate rates; no exact joint rate is inferred.",
            "inferential_status": "All screens are descriptive diagnostics, not formal hypothesis tests, TOST, or global model-adequacy certification."
        },
        "rows": rows,
        "key_findings": [
            "A conventional single-stat PPC interval can contain the observed TV95 in 100% of datasets under the deliberately over-dispersed scenario E, so interval containment alone is not evidence of calibrated spread.",
            "Under scenario C, the same TV95 predictive interval coverage can be 100% while both TV95 and TVmean practical-margin screens fail in every dataset.",
            "Under scenario D, TV95 reference-band and interval checks can pass often despite a shifted center; the TVmean practical-margin screen has zero pass rate in the archived cells, providing a complementary diagnostic.",
            "Under scenario B, the transition-JS violation rate is 100%, illustrating that marginal imbalance diagnostics do not substitute for order-sensitive checks.",
            "At N=96, the TV95 practical-margin screen has low pass rates even under the true generator (0–1.7% in the archived cells). It must therefore be described as a stringent precision screen, not a calibrated accept/reject test."
        ]
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    csv_path = a.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"output": str(a.output), "csv": str(csv_path), "rows": len(rows)}, indent=2))

if __name__ == "__main__": main()
