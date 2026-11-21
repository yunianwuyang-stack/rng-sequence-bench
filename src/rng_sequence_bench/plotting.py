"""Optional display helpers for model-independent sequence audits."""


def plot_repeat_check(report, ax=None):
    """Plot observed and forecast repeat rates on fitting and held-out histories."""
    if ax is None:
        from matplotlib import pyplot as plt

        _, ax = plt.subplots()
    diagnostics = report["observed_history_repetition"]
    for key, label in (("observed", "Observed"), ("predicted", "Forecast")):
        ax.plot([0, 1], [100 * diagnostics[part][key] for part in ("fit", "holdout")],
                marker="o", label=label)
    ax.set(xticks=[0, 1], xticklabels=["Fit window", "Holdout window"], ylabel="Repeat rate (%)")
    ax.legend(frameon=False)
    return ax
