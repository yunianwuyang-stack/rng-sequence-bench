args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
draws <- read.csv(args[1], check.names = FALSE, colClasses = "numeric")
required <- c("chain", "draw", "divergent__", "treedepth__", "energy__")
excluded <- c(required, "stepsize__", "accept_stat__", "n_leapfrog__")
depth <- as.integer(args[3])
stopifnot(all(required %in% names(draws)), !anyDuplicated(names(draws)),
          all(is.finite(as.matrix(draws))), all(draws$chain >= 1),
          all(draws$chain == floor(draws$chain)), all(draws$divergent__ %in% 0:1),
          all(draws$treedepth__ %in% seq.int(0L, depth)))
chains <- sort(unique(draws$chain))
parts <- lapply(chains, function(chain) {
    rows <- draws[draws$chain == chain, ]
    rows <- rows[order(rows$draw), ]
    stopifnot(identical(rows$draw, as.numeric(seq_len(nrow(rows)))))
    rows
})
counts <- vapply(parts, nrow, integer(1))
parameters <- setdiff(names(draws), excluded)
stopifnot(length(chains) >= 2L, length(unique(counts)) == 1L,
          all(counts >= 4L), length(parameters) >= 1L)

chain_diags <- do.call(rbind, lapply(parts, function(rows) {
    variance <- var(rows$energy__)
    data.frame(chain = rows$chain[1L], retained = nrow(rows), divergences = sum(rows$divergent__),
               at_max_treedepth = sum(rows$treedepth__ == depth),
               e_bfmi = if (variance > 0) mean(diff(rows$energy__)^2) / variance else NA_real_)
}))
posterior_diags <- do.call(rbind, lapply(parameters, function(variable) {
    values <- vapply(parts, function(rows) rows[[variable]], numeric(counts[1L]))
    data.frame(parameter = variable, mean = mean(values),
               q025 = unname(quantile(values, .025)), q975 = unname(quantile(values, .975)),
               rank_rhat = posterior::rhat(values), bulk_ess = posterior::ess_bulk(values),
               tail_ess = posterior::ess_tail(values), mean_mcse = posterior::mcse_mean(values))
}))
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
write.csv(chain_diags, file.path(args[2], "chain_diagnostics.csv"), row.names = FALSE)
write.csv(posterior_diags, file.path(args[2], "posterior_diagnostics.csv"), row.names = FALSE)
