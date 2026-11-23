"""Cross-language numerical parity tests (independent reference & optional R cross-check)."""

import shutil
import subprocess

import numpy as np
from scipy.spatial.distance import jensenshannon

from rng_sequence_bench import forecast_diagnostics, from_ragged, sequence_diagnostics


def test_exact_reference_parity_ragged_and_nonuniform():
    """Verify exact numerical agreement against independent scalar definitions."""
    ragged = [
        np.array([0, 1, 2, 0, 1, 2, 1, 0], dtype=int),
        np.array([2, 2, 1, 0, 1, 2], dtype=int),
        np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=int),
    ]
    y, mask = from_ragged(ragged)
    n, max_t = y.shape
    k = 3
    ref_q = np.array([0.5, 0.3, 0.2])

    rng = np.random.default_rng(2026)
    probs = rng.dirichlet([1.5, 2.0, 1.0], size=(n, max_t - 1))
    # Independent scalar calculation of log loss & multicategory Brier on unmasked suffix targets
    ll_vals, brier_vals = [], []
    for i, seq in enumerate(ragged):
        for t in range(1, len(seq)):
            p = probs[i, t - 1]
            c = seq[t]
            ll_vals.append(-np.log(p[c]))
            onehot = np.zeros(k)
            onehot[c] = 1.0
            brier_vals.append(float(np.sum((p - onehot) ** 2)))

    fdiag = forecast_diagnostics(probs, y[:, 1:], mask=mask[:, 1:])
    assert np.isclose(fdiag["named_scores"]["log_loss"], np.mean(ll_vals), atol=1e-12)
    assert np.isclose(fdiag["named_scores"]["brier"], np.mean(brier_vals), atol=1e-12)

    # Independent scalar calculation of per-participant TV against non-uniform reference q
    tv_ref = []
    for seq in ragged:
        props = np.array([(seq == j).mean() for j in range(k)])
        tv_ref.append(0.5 * np.abs(props - ref_q).sum())
    sdiag = sequence_diagnostics(y, categories=k, reference=ref_q, mask=mask)
    assert np.isclose(sdiag["tv_mean"], np.mean(tv_ref), atol=1e-12)

    # Optional live Rscript parity verification when R is installed on PATH
    if shutil.which("Rscript") is not None:
        r_code = (
            "p <- c(0.5, 0.3, 0.2); q <- c(0.5, 0.3, 0.2); "
            "y <- c(1, 2, 3, 1, 2, 3, 2, 1); "
            "props <- sapply(1:3, function(j) mean(y == j)); "
            "tv <- 0.5 * sum(abs(props - q)); "
            "ll <- mean(-log(p[y[-1]])); "
            "brier <- mean(sapply(y[-1], function(c) { e <- rep(0,3); e[c] <- 1; sum((p - e)^2) })); "
            "rep_rate <- mean(diff(y) == 0); "
            "cat(sprintf('%.12f %.12f %.12f %.12f\\n', tv, ll, brier, rep_rate))"
        )
        out = subprocess.check_output(["Rscript", "-e", r_code], text=True).strip()
        r_tv, r_ll, r_brier, r_rep = [float(x) for x in out.split()]
        py_sdiag = sequence_diagnostics(ragged[0][None, :], categories=3, reference=ref_q)
        py_fdiag = forecast_diagnostics(
            np.broadcast_to(ref_q, (1, len(ragged[0]) - 1, 3)).copy(),
            ragged[0][None, 1:],
        )
        assert np.isclose(r_tv, py_sdiag["tv_mean"], atol=1e-10)
        assert np.isclose(r_rep, py_sdiag["repeat_mean"], atol=1e-10)
        assert np.isclose(r_ll, py_fdiag["named_scores"]["log_loss"], atol=1e-10)
        assert np.isclose(r_brier, py_fdiag["named_scores"]["brier"], atol=1e-10)
