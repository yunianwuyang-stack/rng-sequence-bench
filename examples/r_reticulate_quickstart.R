# Pass R choices and simulated suffixes to the installed Python audit package.
library(reticulate)
rsb <- import("rng_sequence_bench", convert=FALSE)
set.seed(2026)
n <- 10L; total <- 40L; cut <- 25L; k <- 4L
choices <- matrix(sample.int(k,n*total,replace=TRUE)-1L,nrow=n)
forecasts <- array(1/k,c(n,total-1L,k))
pairs <- lapply(seq_len(16L),function(i) {
    suffix <- matrix(sample.int(k,n*(total-cut),replace=TRUE)-1L,nrow=n)
    tuple(r_to_py(suffix),r_to_py(suffix))
})
report <- rsb$audit_sequence_contexts(r_to_py(choices),NULL,r_to_py(forecasts),
                                      r_to_py(pairs),fit_through=cut,initial_transitions=4L)
scores <- py_to_r(report$get("holdout_scores_named"))
stopifnot(abs(scores$log_loss$mean-log(k)) < 1e-10)
print(scores$log_loss)
