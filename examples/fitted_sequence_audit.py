"""Fit a nonuniform synthetic model and audit its forecasts, two starts, and custom stats."""

import argparse
import json
from pathlib import Path

import numpy as np
from rng_sequence_bench import (
    audit_sequence_contexts,
    check_forecast_causality,
    fixed_count_test,
    plot_repeat_check,
)
from rng_sequence_bench.choice_models import (
    RD,
    design,
    design_features,
    fit_batch,
    predict,
    simulate,
)


def alternation_rate(sequences, mask=None):
    """Domain-specific statistic: share of adjacent trials that switch categories."""
    switched = sequences[:, 1:] != sequences[:, :-1]
    if mask is None:
        return float(switched.mean())
    valid = mask[:, 1:] & mask[:, :-1]
    return float(switched[valid].mean())


def fitted_audit():
    rng = np.random.default_rng(47023)
    coords = np.arange(3)[:, None]
    truth = np.tile([-1.1, -0.45], (12, 1))
    y = simulate(truth, RD, coords, rng.random((12, 60)))
    cut = 40
    prefix = y[:, :cut]

    # BEGIN ARTICLE EXAMPLE
    # Step 1: Fit a conditional-logit model on each participant's prefix (trials 1..40).
    x = design(design_features(prefix, coords, RD), RD)
    beta = fit_batch(x, prefix[:, 1:])

    def forecast(history):
        x = design(design_features(history, coords, RD), RD)
        return predict(x, beta)

    # Step 2: Verify that holdout forecasts do not depend on future responses.
    check = check_forecast_causality(forecast, y, cut, len(coords))

    # Step 3: Generate paired reset-start and prefix-continuation suffixes with shared uniforms.
    uniforms = rng.random((32, *y.shape))
    pairs = (
        (
            simulate(beta, RD, coords, u[:, cut:]),
            simulate(beta, RD, coords, u, prefix=prefix)[:, cut:],
        )
        for u in uniforms
    )

    # Step 4: Audit forecasts, calibration, paired starts, and a domain-specific custom stat.
    report = audit_sequence_contexts(
        y,
        coords,
        forecast(y),
        pairs,
        fit_through=cut,
        custom_stats={"alternation_rate": alternation_rate},
    )
    reference = fixed_count_test(y, len(coords), permutations=499)
    # END ARTICLE EXAMPLE

    assert check["altered_trials"] == [41, 51]
    assert report["replicates"] == 32 and report["suffix_length"] == 20
    assert report["holdout_forecasts"]["named_scores"]["log_loss"] < np.log(3)
    assert "alternation_rate" in report["observed_prefix"]["custom_statistics"]
    return {
        "seed": 47023,
        "causality": check,
        "audit": report,
        "fixed_count": reference,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, help="Write full machine-readable JSON results")
    p.add_argument("--plot", type=Path, help="Write a diagnostic plot")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print the complete JSON report instead of the concise article summary",
    )
    args = p.parse_args()
    report = fitted_audit()
    evidence = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.write_text(evidence, encoding="utf-8")
    elif args.verbose:
        print(evidence, end="")
    else:
        named = {
            k: round(v, 4)
            for k, v in report["audit"]["holdout_forecasts"]["named_scores"].items()
        }
        alt = report["audit"]["observed_prefix"]["custom_statistics"]["alternation_rate"]
        print(f"Causality check: {report['causality']}")
        print(f"Holdout scores:  {named}")
        print(
            f"Custom stat (alternation_rate): obs={alt['observed']:.4f}, "
            f"sim_median={alt['median']:.4f} "
            f"[{alt['predictive95'][0]:.4f}, {alt['predictive95'][1]:.4f}]"
        )
    if args.plot:
        plot_repeat_check(report["audit"]).figure.savefig(args.plot, bbox_inches="tight")


if __name__ == "__main__":
    main()
