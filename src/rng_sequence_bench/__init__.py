"""Reproducible benchmarks for finite categorical sequences."""

from rng_sequence_bench.analysis import (
    benchmark,
    fixed_count_test,
    pooling_benchmark,
    transfer_benchmark,
)
from rng_sequence_bench.chronological import standardized_benchmark
from rng_sequence_bench.contexts import (
    audit_sequence_contexts,
    check_forecast_causality,
    check_prefix_preprocessing,
    rolling_forecasts,
)
from rng_sequence_bench.data import from_ragged, read_ragged_trials, read_trials
from rng_sequence_bench.evaluation import (
    debiased_ece_l2,
    ece_binning_sensitivity,
    forecast_diagnostics,
)
from rng_sequence_bench.hierarchy import run_hierarchy_chain
from rng_sequence_bench.plotting import (
    plot_forecast_vs_generation,
    plot_horizon_curve,
    plot_reliability_diagram,
    plot_repeat_check,
    plot_sbc_ecdf_diff,
)
from rng_sequence_bench.predictive import (
    adequacy_set,
    mc_ranking_probability,
    posterior_predictive_checks,
    reference_band_screen,
)
from rng_sequence_bench.recovery import recovery_benchmark
from rng_sequence_bench.sequence_metrics import sequence_diagnostics

__all__ = [
    "adequacy_set",
    "audit_sequence_contexts",
    "benchmark",
    "check_forecast_causality",
    "check_prefix_preprocessing",
    "debiased_ece_l2",
    "ece_binning_sensitivity",
    "fixed_count_test",
    "forecast_diagnostics",
    "from_ragged",
    "mc_ranking_probability",
    "plot_forecast_vs_generation",
    "plot_horizon_curve",
    "plot_reliability_diagram",
    "plot_repeat_check",
    "plot_sbc_ecdf_diff",
    "pooling_benchmark",
    "posterior_predictive_checks",
    "read_ragged_trials",
    "read_trials",
    "recovery_benchmark",
    "reference_band_screen",
    "rolling_forecasts",
    "run_hierarchy_chain",
    "sequence_diagnostics",
    "standardized_benchmark",
    "transfer_benchmark",
]
