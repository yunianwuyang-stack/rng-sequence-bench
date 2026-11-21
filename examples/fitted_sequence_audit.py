"""Fit a nonuniform synthetic model and audit its forecasts and two starts."""

import argparse
import json
from pathlib import Path

import numpy as np
from rng_sequence_bench import audit_sequence_contexts, check_forecast_causality, fixed_count_test, plot_repeat_check
from rng_sequence_bench.choice_models import RD, design, design_features, fit_batch, predict, simulate


def fitted_audit():
    rng = np.random.default_rng(47023)
    coords = np.arange(3)[:, None]
    truth = np.tile([-1.1, -.45], (12, 1))
    y = simulate(truth, RD, coords, rng.random((12, 60)))
    cut = 40
    prefix = y[:, :cut]

    # BEGIN ARTICLE EXAMPLE
    x = design(design_features(prefix, coords, RD), RD)
    beta = fit_batch(x, prefix[:, 1:])

    def forecast(history):
        x = design(design_features(history, coords, RD), RD)
        return predict(x, beta)

    check = check_forecast_causality(forecast, y, cut, len(coords))
    uniforms = rng.random((32, *y.shape))
    pairs = (
        (simulate(beta, RD, coords, u[:, cut:]),
         simulate(beta, RD, coords, u, prefix=prefix)[:, cut:])
        for u in uniforms
    )
    report = audit_sequence_contexts(
        y, coords, forecast(y), pairs, fit_through=cut,
    )
    reference = fixed_count_test(y, len(coords), permutations=499)
    # END ARTICLE EXAMPLE

    assert check["altered_trials"] == [41, 51]
    assert report["replicates"] == 32 and report["suffix_length"] == 20
    assert report["holdout_forecasts"]["scores"][0] < np.log(3)
    return {"seed":47023,"causality":check,"audit":report,"fixed_count":reference}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, help="Write machine-readable results")
    p.add_argument("--plot", type=Path, help="Write a diagnostic plot")
    args = p.parse_args()
    report = fitted_audit()
    evidence = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.write_text(evidence, encoding="utf-8")
    else:
        print(evidence, end="")
    if args.plot:
        plot_repeat_check(report["audit"]).figure.savefig(args.plot, bbox_inches="tight")


if __name__ == "__main__":
    main()
