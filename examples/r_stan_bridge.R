# Minimal R -> Python bridge example for RNG Sequence Bench (v0.9.2)
# Demonstrates exporting model forecasts and paired simulated suffixes from R
# (e.g., fitted with rstan, brms, cmdstanr, or samplr) for auditing in Python.
#
# Run:
#   Rscript examples/r_stan_bridge.R work_r_bridge
#   python examples/audit_from_r_export.py work_r_bridge

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else tempdir()
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(2026)
N <- 10; T_len <- 40; K <- 4; cut <- 25; R_rep <- 16

# 1. Observed choices (zero-based integers 0..K-1)
choices <- matrix(sample(0:(K - 1), N * T_len, replace = TRUE), nrow = N, ncol = T_len)
write.csv(choices, file.path(out_dir, "choices.csv"), row.names = FALSE)

# 2. One-step-ahead forecast probabilities from any R/Stan model: N x (T-1) x K
#    Saved as a flat table with columns subject, trial, p0..p{K-1}
probs_df <- expand.grid(subject = seq_len(N), trial = 2:T_len)
for (k in seq_len(K)) probs_df[[paste0("p", k - 1)]] <- 1 / K
write.csv(probs_df, file.path(out_dir, "forecasts.csv"), row.names = FALSE)

# 3. Paired reset-start and prefix-continuation simulated suffixes (R x N x (T - cut))
suffix_len <- T_len - cut
for (r in seq_len(R_rep)) {
  reset_s <- matrix(sample(0:(K - 1), N * suffix_len, replace = TRUE), nrow = N)
  pref_s  <- matrix(sample(0:(K - 1), N * suffix_len, replace = TRUE), nrow = N)
  write.csv(reset_s, file.path(out_dir, sprintf("reset_%03d.csv", r)), row.names = FALSE)
  write.csv(pref_s,  file.path(out_dir, sprintf("prefix_%03d.csv", r)), row.names = FALSE)
}
cat("Exported R/Stan audit inputs to:", out_dir, "\n")
