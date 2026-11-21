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
)
from rng_sequence_bench.evaluation import forecast_diagnostics
from rng_sequence_bench.data import read_trials
from rng_sequence_bench.hierarchy import run_hierarchy_chain
from rng_sequence_bench.predictive import posterior_predictive_checks
from rng_sequence_bench.recovery import recovery_benchmark
from rng_sequence_bench.sequence_metrics import sequence_diagnostics

__all__ = [
    "audit_sequence_contexts",
    "benchmark",
    "check_forecast_causality",
    "fixed_count_test",
    "forecast_diagnostics",
    "pooling_benchmark",
    "posterior_predictive_checks",
    "read_trials",
    "recovery_benchmark",
    "run_hierarchy_chain",
    "sequence_diagnostics",
    "standardized_benchmark",
    "transfer_benchmark",
]
