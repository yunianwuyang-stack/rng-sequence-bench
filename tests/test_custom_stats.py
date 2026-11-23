"""Contract tests for user-supplied domain-specific custom_stats callbacks."""

import numpy as np
import pytest

from rng_sequence_bench import (
    audit_sequence_contexts,
    from_ragged,
    posterior_predictive_checks,
    sequence_diagnostics,
)


def _adv_share(sequences, mask=None):
    """Advantageous deck share (categories 2 and 3 in a 4-category task)."""
    adv = sequences >= 2
    if mask is None:
        return float(adv.mean())
    return float((adv & mask).sum() / mask.sum())


def _stay_rate_no_mask(sequences):
    """One-argument custom statistic without mask parameter."""
    return float((sequences[:, 1:] == sequences[:, :-1]).mean())


def _category_means(sequences, mask=None):
    """Vector-valued custom statistic returning share of each category."""
    k = 4
    if mask is None:
        return [(sequences == c).mean() for c in range(k)]
    denom = mask.sum()
    return [float(((sequences == c) & mask).sum() / denom) for c in range(k)]


def test_sequence_diagnostics_custom_stats_equal_and_ragged():
    rng = np.random.default_rng(101)
    y = rng.integers(0, 4, size=(8, 20))
    rep = sequence_diagnostics(
        y,
        categories=4,
        custom_stats={"adv_share": _adv_share, "stay": _stay_rate_no_mask},
    )
    assert "custom_statistics" in rep
    assert np.isclose(rep["custom_statistics"]["adv_share"], (y >= 2).mean())
    assert np.isclose(rep["custom_statistics"]["stay"], (y[:, 1:] == y[:, :-1]).mean())

    ragged = [rng.integers(0, 4, size=L) for L in (12, 16, 20, 15)]
    padded, mask = from_ragged(ragged)
    rep_m = sequence_diagnostics(
        padded,
        categories=4,
        mask=mask,
        custom_stats={"adv_share": _adv_share, "shares": _category_means},
    )
    expected_adv = sum((r >= 2).sum() for r in ragged) / sum(len(r) for r in ragged)
    assert np.isclose(rep_m["custom_statistics"]["adv_share"], expected_adv)
    assert np.isclose(sum(rep_m["custom_statistics"]["shares"]), 1.0)


def test_ppc_and_audit_contexts_custom_stats_with_mask():
    rng = np.random.default_rng(202)
    ragged = [rng.integers(0, 4, size=L) for L in (16, 20, 24, 18, 22, 16)]
    y, mask = from_ragged(ragged)
    n, max_t = y.shape
    cut = 10
    probs = np.full((n, max_t - 1, 4), 0.25)
    # Build paired replicates where prefix arm has higher advantageous share
    pairs = []
    for _ in range(12):
        reset_s = rng.integers(0, 4, size=(n, max_t - cut))
        pref_s = rng.choice([0, 1, 2, 3], size=(n, max_t - cut), p=[0.1, 0.1, 0.4, 0.4])
        pairs.append((reset_s, pref_s))

    ppc = posterior_predictive_checks(
        y[:, cut:],
        [p[1] for p in pairs],
        categories=4,
        mask=mask[:, cut:],
        custom_stats={"adv_share": _adv_share, "shares": _category_means},
    )
    assert "adv_share" in ppc["custom_statistics"]
    assert "custom_adv_share_abs_diff" in ppc["distance_from_observed"]
    assert "custom_shares_abs_diff" not in ppc["distance_from_observed"]
    assert ppc["custom_statistics"]["adv_share"]["median"] > 0.65

    audit = audit_sequence_contexts(
        y,
        None,
        probs,
        pairs,
        fit_through=cut,
        initial_transitions=3,
        mask=mask,
        custom_stats={"adv_share": _adv_share},
    )
    assert "custom_statistics" in audit["uniform_start"]
    assert "custom_statistics" in audit["observed_prefix"]
    assert "adv_share" in audit["uniform_minus_prefix_custom"]
    # Reset (~0.50) minus prefix (~0.80) should be clearly negative
    assert audit["uniform_minus_prefix_custom"]["adv_share"]["median"] < -0.15


def test_custom_stats_validation_errors():
    y = np.zeros((3, 8), dtype=int)
    with pytest.raises(ValueError, match="mapping"):
        sequence_diagnostics(y, categories=2, custom_stats=[_stay_rate_no_mask])
    with pytest.raises(ValueError, match="non-empty"):
        sequence_diagnostics(y, categories=2, custom_stats={" ": _stay_rate_no_mask})
    with pytest.raises(ValueError, match="callable"):
        sequence_diagnostics(y, categories=2, custom_stats={"bad": 123})
    with pytest.raises(ValueError, match="non-finite"):
        sequence_diagnostics(y, categories=2, custom_stats={"nan_stat": lambda s: float("nan")})
