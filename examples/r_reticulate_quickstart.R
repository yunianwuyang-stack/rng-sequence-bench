# Direct R -> RNG Sequence Bench (v0.9.0) quickstart via reticulate
library(reticulate)
rsb <- import("rng_sequence_bench")
np  <- import("numpy")

# Observed choices (N=4, T=12, K=3) and uniform forecast probabilities
y <- np$array(matrix(sample(0:2, 48, replace = TRUE), nrow = 4L), dtype = "int64")
p <- np$full(tuple(4L, 11L, 3L), 1 / 3)
pairs <- lapply(1:8, function(i) tuple(y[, 9:12], y[, 9:12]))

# Custom domain statistic evaluated alongside built-in sequence summaries
win_switch <- function(seq) mean(seq[, -1, drop = FALSE] != seq[, -ncol(seq), drop = FALSE])
rep <- rsb$audit_sequence_contexts(y, NULL, p, pairs, fit_through = 8L,
                                   custom_stats = list(switch_rate = win_switch))
print(rep$holdout_forecasts$named_scores)
