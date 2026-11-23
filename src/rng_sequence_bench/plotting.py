"""Publication-ready diagnostic display helpers for sequence audits."""

import numpy as np


def _get_ax(ax, figsize=(5.2, 3.6)):
    if ax is not None:
        return ax
    from matplotlib import pyplot as plt

    _, created = plt.subplots(figsize=figsize)
    return created


def plot_repeat_check(report, ax=None):
    """Plot observed and forecast repeat rates on fitting and held-out histories."""
    ax = _get_ax(ax, figsize=(4.6, 3.3))
    diagnostics = report["observed_history_repetition"]
    for key, label in (("observed", "Observed"), ("predicted", "Forecast")):
        ax.plot(
            [0, 1],
            [100 * diagnostics[part][key] for part in ("fit", "holdout")],
            marker="o",
            label=label,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Fit window", "Holdout window"],
        ylabel="Repeat rate (%)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return ax


def plot_forecast_vs_generation(
    models,
    observed_value,
    observed_ci=None,
    *,
    xlabel="Held-out log loss (nats/choice)",
    ylabel="Generated summary statistic",
    title=None,
    ax=None,
):
    """Plot held-out forecast loss against generated summary with reference band.

    ``models`` maps display labels to dicts containing ``log_loss``,
    ``gen_median``, and optional ``log_loss_ci95`` and ``gen_ci95``.
    """
    ax = _get_ax(ax)
    if observed_ci is not None:
        ax.axhspan(
            float(observed_ci[0]),
            float(observed_ci[1]),
            color="#303030",
            alpha=0.12,
            label="Observed 95% reference band",
            zorder=0,
        )
    ax.axhline(
        float(observed_value),
        color="#222222",
        ls="--",
        lw=1.1,
        label="Observed",
        zorder=1,
    )
    palette = ("#1f5f8b", "#c54d33", "#1b8a78", "#6d599b", "#b87414", "#555555")
    for idx, (label, entry) in enumerate(models.items()):
        x = float(entry["log_loss"])
        y = float(entry["gen_median"])
        xerr = None
        if "log_loss_ci95" in entry and entry["log_loss_ci95"] is not None:
            lo, hi = entry["log_loss_ci95"]
            xerr = [[max(0.0, x - float(lo))], [max(0.0, float(hi) - x)]]
        yerr = None
        if "gen_ci95" in entry and entry["gen_ci95"] is not None:
            lo, hi = entry["gen_ci95"]
            yerr = [[max(0.0, y - float(lo))], [max(0.0, float(hi) - y)]]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=4.8,
            capsize=2.5,
            lw=1.2,
            color=palette[idx % len(palette)],
            label=label,
            zorder=4,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_horizon_curve(
    horizons,
    observed,
    model_curves,
    *,
    observed_ci=None,
    observed_label="Observed",
    xlabel="Simulation horizon t (trials)",
    ylabel="Summary statistic",
    title=None,
    ax=None,
):
    """Plot a sequence summary statistic across increasing simulation horizons."""
    ax = _get_ax(ax)
    ts = np.asarray(horizons, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if observed_ci is not None:
        ci = np.asarray(observed_ci, dtype=float)
        ax.fill_between(ts, ci[:, 0], ci[:, 1], color="#303030", alpha=0.14, zorder=1)
    ax.plot(ts, obs, color="#222222", ls="--", lw=1.5, marker="s", ms=3.2, label=observed_label, zorder=5)
    palette = ("#1f5f8b", "#c54d33", "#6d599b", "#1b8a78", "#b87414")
    for idx, (label, spec) in enumerate(model_curves.items()):
        med = np.asarray(spec["median"], dtype=float)
        c = palette[idx % len(palette)]
        if "ci95" in spec and spec["ci95"] is not None:
            lo, hi = np.asarray(spec["ci95"][0], dtype=float), np.asarray(spec["ci95"][1], dtype=float)
            ax.fill_between(ts, lo, hi, color=c, alpha=0.12, zorder=2)
        ax.plot(ts, med, color=c, lw=1.4, marker="o", ms=3.0, label=label, zorder=4)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_reliability_diagram(calibration_report, *, title="Top-choice reliability", ax=None):
    """Plot nonempty confidence bins against empirical accuracy with the diagonal reference."""
    ax = _get_ax(ax, figsize=(4.5, 4.0))
    bins = [b for b in calibration_report["top_bins"] if b["n"] > 0]
    preds = [float(b["predicted"]) for b in bins]
    obs = [float(b["observed"]) for b in bins]
    ax.plot([0, 1], [0, 1], color="#666666", ls="--", lw=1.0, label="Perfect calibration")
    ax.plot(preds, obs, color="#1f5f8b", marker="o", lw=1.4, label="Empirical")
    ece = calibration_report.get("top_ece")
    corp = (calibration_report.get("top_corp") or {}).get("corp_ece")
    note = []
    if ece is not None:
        note.append(f"ECE = {ece:.3f}")
    if corp is not None:
        note.append(f"CORP = {corp:.3f}")
    if note:
        ax.text(
            0.05,
            0.88,
            "\n".join(note),
            transform=ax.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted confidence",
        ylabel="Observed top-choice accuracy",
        title=title,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    return ax


def plot_sbc_ecdf_diff(
    ranks,
    max_rank,
    *,
    confidence=0.95,
    n_sim=2000,
    seed=2026,
    title="SBC ECDF difference",
    ax=None,
):
    """Plot scaled rank ECDF differences with simultaneous uniform bands.

    Implements the graphical uniformity diagnostic of Sailynoja, Burkner,
    and Vehtari (2022) via Monte Carlo simultaneous Kolmogorov envelopes.
    """
    ax = _get_ax(ax, figsize=(4.8, 3.6))
    arr = np.asarray(ranks, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("ranks must be a non-empty 1D sequence")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    scaled = np.sort(arr / float(max_rank))
    n = len(scaled)
    # Evaluate standardized ECDF deviation on the strict interior of (0, 1) to avoid
    # zero-variance endpoint spikes (Sailynoja et al., 2022).
    grid = np.linspace(1.0 / (max_rank + 2), 1.0 - 1.0 / (max_rank + 2), min(max_rank, 193))
    ecdf_vals = np.searchsorted(scaled, grid, side="right") / n
    diff = ecdf_vals - grid

    rng = np.random.default_rng(seed)
    u = np.sort(rng.integers(0, max_rank + 1, size=(n_sim, n)) / float(max_rank), axis=1)
    sim_ecdf = np.array([np.searchsorted(row, grid, side="right") / n for row in u])
    sim_diff = sim_ecdf - grid[None, :]
    scale = np.sqrt(grid * (1.0 - grid))
    sup_z = np.max(np.abs(sim_diff) / scale[None, :], axis=1)
    crit = float(np.quantile(sup_z, confidence))
    env = np.minimum(1.0 - grid, np.maximum(grid, crit * scale))
    env = crit * scale
    ax.fill_between(
        grid,
        -env,
        env,
        color="#1f5f8b",
        alpha=0.14,
        label=f"{int(round(100 * confidence))}% simultaneous band",
        zorder=1,
    )
    ax.axhline(0.0, color="#666666", ls="--", lw=1.0, zorder=2)
    ax.step(grid, diff, where="post", color="#1f5f8b", lw=1.4, label="Observed ECDF diff", zorder=3)
    ax.set(xlabel="Scaled rank z", ylabel="ECDF(z) - z", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    return ax
