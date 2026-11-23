"""Tests for variable-length sequences and missing-trial masks (v0.8.0)."""

import numpy as np
import pytest

from rng_sequence_bench import (
    audit_sequence_contexts,
    forecast_diagnostics,
    from_ragged,
    posterior_predictive_checks,
    read_ragged_trials,
    sequence_diagnostics,
)


def test_from_ragged_and_masked_sequence_diagnostics_match_unpadded():
    rng = np.random.default_rng(123)
    s0 = rng.integers(4, size=30)
    s1 = rng.integers(4, size=20)
    choices, mask = from_ragged([s0, s1])
    assert choices.shape == (2, 30)
    assert mask[0].all()
    assert mask[1, :20].all() and not mask[1, 20:].any()

    summary = sequence_diagnostics(choices, categories=4, mask=mask)
    # Verify exact marginal TV against individually computed unpadded TVs
    tv0 = np.abs(np.bincount(s0, minlength=4) / len(s0) - 0.25).sum() / 2
    tv1 = np.abs(np.bincount(s1, minlength=4) / len(s1) - 0.25).sum() / 2
    assert np.isclose(summary["tv_mean"], 0.5 * (tv0 + tv1))

    # Verify repeat rate matches unpadded per-person mean
    r0 = (s0[1:] == s0[:-1]).mean()
    r1 = (s1[1:] == s1[:-1]).mean()
    assert np.isclose(summary["repeat_mean"], 0.5 * (r0 + r1))
    assert "participant_conditional_entropy_median" in summary
    assert len(summary["participant_conditional_entropy_iqr"]) == 2


def test_masked_forecast_diagnostics_ignores_padded_trials():
    rng = np.random.default_rng(456)
    y_full = rng.integers(3, size=(2, 10))
    p_full = np.full((2, 10, 3), 1.0 / 3.0)
    mask = np.ones((2, 10), dtype=bool)
    mask[1, 6:] = False

    y_padded = y_full.copy()
    y_padded[1, 6:] = -1
    p_padded = p_full.copy()
    p_padded[1, 6:] = np.nan  # Should be ignored on masked entries

    diag = forecast_diagnostics(p_padded, y_padded, mask=mask)
    assert np.isclose(diag["named_scores"]["log_loss"], np.log(3))
    assert np.isclose(diag["named_scores"]["brier"], 2.0 / 3.0)
    assert np.isclose(diag["named_scores"]["top1_accuracy"], 1.0 / 3.0)


def test_masked_audit_sequence_contexts_variable_suffix_lengths():
    rng = np.random.default_rng(789)
    seqs = [rng.integers(4, size=t) for t in (25, 30, 22, 30)]
    choices, mask = from_ragged(seqs)
    n, max_t = choices.shape
    cut = 15
    probs = np.full((n, max_t - 1, 4), 0.25)
    pairs = [
        (rng.integers(4, size=(n, max_t - cut)), rng.integers(4, size=(n, max_t - cut)))
        for _ in range(8)
    ]
    report = audit_sequence_contexts(
        choices, None, probs, pairs, fit_through=cut, initial_transitions=3, seed=42, mask=mask
    )
    assert np.isclose(report["holdout_scores_named"]["log_loss"]["mean"], np.log(4))
    assert report["uniform_start"]["replicates"] == 8
    assert report["observed_prefix"]["replicates"] == 8

    # Also test direct posterior_predictive_checks with mask
    ppc = posterior_predictive_checks(
        choices[:, cut:], [p[0] for p in pairs], categories=4, mask=mask[:, cut:]
    )
    assert "participant_conditional_entropy_median" in ppc["statistics"]


def test_read_ragged_trials_with_missing_tokens(tmp_path):
    csv_path = tmp_path / "ragged.csv"
    csv_path.write_text(
        "subject,trial,choice\n"
        "p1,1,A\np1,2,B\np1,3,NA\np1,4,C\n"
        "p2,1,C\np2,2,A\n",
        encoding="utf-8",
    )
    choices, mask = read_ragged_trials(csv_path, ["A", "B", "C"])
    assert choices.shape == (2, 4)
    assert choices[0].tolist() == [0, 1, -1, 2]
    assert mask[0].tolist() == [True, True, False, True]
    assert choices[1].tolist() == [2, 0, -1, -1]
    assert mask[1].tolist() == [True, True, False, False]
