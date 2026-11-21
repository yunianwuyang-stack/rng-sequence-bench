# RNG Sequence Bench

Compare marginal balance, ordering beyond category counts, and the prediction of human categorical sequences. The library accepts equal-length, independently sampled participant sequences and a finite alphabet with known coordinates. It can also use ordinal coordinates when an outcome ordering is meaningful.

## Install

With Python 3.11 or later, install from this directory.

```bash
python -m pip install .
```

For development with uv, run `uv sync` and `uv run pytest`.

For a published version, use the wheel in the [release assets](https://github.com/yunianwuyang-stack/rng-sequence-bench/releases).

## Run

Supply a CSV with `subject`, `trial`, and `choice`. Trial numbers start at 1 for each subject and have no gaps. Use one sequence per independent participant. Category labels and coordinate rows follow the same specified order. For example, for `1,2,3` on a line, omit `--coordinates` or supply `[[0],[1],[2]]`. For a grid, pass the actual 2-D positions. Here each sequence must have at least 200 trials.

```bash
rng-sequence-bench trials.csv results.json --alphabet 1,2,3 \
  --origins 100,125,150,175 --width 25 --simulations 40
```

The package also exposes `fixed_count_test` and `benchmark` for NumPy arrays. Encode the alphabet as integers from zero through the alphabet size minus one. See the function documentation for options.

In Python, `read_trials("trials.csv", ["1", "2", "3"])` reads the same CSV into a zero-based NumPy array ordered by subject and trial.

### Standardized nested forecasts

Add `--standardized --inner-width 25` to run the standardized protocol. This path compares uniform IID, participant-specific frequencies, smoothed first- and second-order Markov, RD, RDC, alternative gap functions, a decaying trace, and frequency terms. It fits each scaler from the inner or outer training prefix only. Hyperparameters are selected from a prior inner window. The reported scores apply to outer observations. Use `standardized_benchmark(choices, coordinates, origins, widths, inner_width=25)` when outer windows have different widths.

### Plug in probabilities or generated sequences

```python
from rng_sequence_bench import forecast_diagnostics, sequence_diagnostics

probability_report = forecast_diagnostics(probabilities, targets)
sequence_report = sequence_diagnostics(generated_choices, coordinates)
```

Forecast arrays have shape (participants, trials, categories). Targets and choices have shape (participants, trials), use zero-based integer categories, and match the coordinate rows. The reports include classwise reliability, diagnostic multiclass slope and centered offsets, a Brier decomposition, and aggregate summaries of total variation, repeats, run lengths, transitions and first-order conditional entropy. Calibration slopes are undefined for constant forecasts or missing observed categories. These cases return a status with null coefficients. For files, save these arrays as named members in an NPZ and run `rng-sequence-checks fit.npz checks.json`. The optional `reference` array is a probability vector for the sequence total-variation calculation.

For a complete matched-history audit, pass an external model's probabilities for choices 2 through the last choice, and an iterable of two-array suffix pairs. Each pair contains a uniform-start replication and a replication initialized with the observed prefix. Keep participant rows, category indices and horizons aligned on both sides. Use one parameter draw or fixed parameter set per pair and coordinate the random streams at the generator. The report includes holdout log loss, Brier loss, top-choice and classwise calibration, repeat residuals on actual histories, and predictive summaries for each start.

```python
from rng_sequence_bench import audit_sequence_contexts, check_forecast_causality

test = check_forecast_causality(forecast, choices, fit_through=140, categories=7)
report = audit_sequence_contexts(
    choices, coordinates, forecast(choices), generated_suffix_pairs,
    fit_through=140,
)
```

The causality check calls the supplied forecast on perturbed suffixes. The caller supplies the frozen fit or uses only the prefix inside the callable. Probabilities that precede a perturbation must be unchanged. Neither function assumes a logit or any other model family. Report generated intervals as predictive intervals for posterior-state draws or simulation intervals for fixed parameters.

Use `transfer_benchmark` when each participant has two matched sessions. It fits both models in the first session and evaluates fixed coefficients in the second, using only the history observed up to each second-session forecast. Align the rows by verified participant identifiers before calling it.

Use `--frequency-window 10` to score cumulative and ten-choice frequency-deficit terms, separately and jointly, plus their joint model with log age. The feature is the expected uniform count minus the observed count, divided by the square root of the number of available prior choices. `--include-pooled` adds shared RD and RDC fits.

The Python function `pooling_benchmark` holds out participants from group fitting. It reports forecasts with fixed group coefficients and forecasts that adapt to each held-out participant's training prefix under a penalty centered on the group fit. The latter is partial pooling on a penalized likelihood scale.

## Methods and output

- `order` reports total variation from uniformity, including its mean and 95th percentile, and the corresponding finite-length IID references. For adjacent repetition, it uses the exact expectation conditional on each sequence's counts and a plus-one Monte Carlo permutation test that shuffles each sequence independently.
- `rolling` scores non-overlapping future windows using prefix-only fits. Coefficients stay fixed in a window and the observed choice history advances one trial at a time. The first response initializes history. It is never a fitted target.
- `participant_bootstrap` and `paired_model_minus_RDC` average over windows within participants and resample participants. The score order is log loss in nats, summed multiclass Brier loss without division by the number of alternatives, and fractional top-choice accuracy. The latter grants equal credit across tied maxima.
- `calibration` reports ten fixed confidence bins and the weighted absolute discrepancy between mean top confidence and tie-aware accuracy. This is **top-choice calibration**, with Brier loss reported separately as a proper score for the full probability vector. Empty bins appear as `null`.
- `repeat_coefficient_decomposition` reports pooled RD and RDC coefficient changes, within-choice-set feature correlations, and modeled repeat probabilities with the fitted repetition term or with that term zeroed while holding the other coefficients fixed. This last comparison is a model perturbation on the observed histories. The binary case omits correlations because repetition and distance are equivalent up to an affine transformation.
- `plug_in_simulation` compares observed sequence summaries with newly generated sequences using the fitted coefficients from the last training prefix. These simulations propagate the generated history forward and start from a uniform draw. Their intervals describe repeated simulations conditional on the fitted parameters. Posterior predictive inference would require posterior draws as an additional input.

RD includes repeat and distance. RDC adds log2 trials since a candidate was last chosen. The decaying-memory model substitutes an exponentially decayed selection trace with a fixed half-life. The directional-momentum model substitutes the cosine between incoming and proposed displacement vectors, with zero alignment for a zero displacement. The combined model adds that term to RDC. All choice models use the same ridge convention. Uniform, participant-specific IID and first-order Markov models provide reference forecasts. The directional model is a discrete comparison motivated by directional persistence, separate from the Hamiltonian recycled-momentum implementation in [Castillo et al. (2024)](https://doi.org/10.1371/journal.pcbi.1011739). The RDC definition follows [Angelike and Musch (2025)](https://doi.org/10.1371/journal.pone.0324870).

The source package contains code and synthetic tests. Public research data are obtained from their original sources under their original terms.

## Native chain diagnostics

With R and the `posterior` package installed, supply a numeric CSV with one row per **retained** draw. Required columns are `chain`, `draw` (consecutive from 1 within a chain), `divergent__`, `treedepth__`, and `energy__`. Append named scalar parameter columns. Rows can be unordered and chains must have equal lengths. Supply the actual fitting limit.

```bash
rng-chain-diagnostics retained.csv diagnostics --max-treedepth 10
```

This writes per-chain divergence counts, tree-depth counts and E-BFMI, plus posterior means, intervals, rank-normalized split R-hat, bulk/tail ESS and mean MCSE. Missing diagnostic values are retained as `NA`.

## License

Apache-2.0 for the software in this directory.
## Recovery and hierarchical generation

```python
from rng_sequence_bench import (
    posterior_predictive_checks,
    recovery_benchmark,
    run_hierarchy_chain,
)

simulated = recovery_benchmark(
    sizes=(24, 96), lengths=(50, 100, 200, 250),
    strengths=(.65, 1.35), replicates=12, workers=8,
)
report = posterior_predictive_checks(observed, replicated_sequences, coordinates)
```

The recovery benchmark implements eight generators on a nine-key grid, including a deliberately out-of-set switching process. It reports coefficient error, sign and rank recovery, local interval coverage, and disagreement among the model winners by next-choice loss and generation criteria. Set coupled_validation=False to use independent uniform streams for the truth and candidate replications. For multiprocessing, call this API from a script protected by the Python main guard.

The generic Gibbs–Metropolis function run_hierarchy_chain accepts a four-dimensional candidate-feature array (participants, training trials, categories, coefficients) and the corresponding integer choices. Supply one positive prior scale per coefficient plus the warmup length, retained count, seed, and a record callback. Use multiple independent chains and check scalar R-hat and ESS. The optional native path points to a locally compiled copy of logit_likelihood.cpp. Without it the sampler uses NumPy. The exact categorical likelihood determines acceptance in both implementations.

For a Normal/HalfNormal/LKJ prior, create a separate `SeparatedPopulation` from `rng_sequence_bench.separated_prior` for each chain. Pass its `update` method as `population_update`. For a sensitivity run, also pass its `interweave` method as `interweave` to update scales or correlation loadings in standardized coordinates. Both options target the same posterior.

posterior_predictive_checks accepts matching populations of observed and already replicated sequences, and the actual category coordinates. It reports repeat, run, recurrence, distance, entropy, transition, and total-variation summaries by posterior replicate. Histograms and observed intervals are stored in machine-readable form. Treat predictive quantiles and tail fractions as descriptive model checks.

## Run the examples from a fresh clone

```bash
python -m pip install .
python examples/independent_choices.py
python examples/matched_contexts.py
```

Both examples use simulated independent choices. The second reads a trial CSV and compares two generated suffixes on an aligned horizon. The three reported scores equal the analytic log loss, Brier loss, and top-choice accuracy for a uniform model.

## Contributing

Open an issue or pull request with a synthetic example that reproduces the requested change. Run the tests with `python -m pytest` before submitting.

## License

The software is licensed under Apache-2.0. See [LICENSE](LICENSE).
## Changes in 0.7.0

For categorical tasks without a spatial layout, call `audit_sequence_contexts` with keyword arguments `probabilities`, `replicated_pairs`, and `fit_through`. Omitting `coordinates` skips the distance summaries and preserves the alphabet from the probability columns, even when a category is absent from the data. For direct simulation summaries, pass `categories=K` to `posterior_predictive_checks`.

`rolling_forecasts(choices, fit_forecast, origins, width, categories=K)` calls a fitting function with the prefix for each origin. The returned forecasting callable supplies probabilities for choices 2 through the end of its input, using only preceding responses. The report keeps origin-specific intervals and records the suffix perturbation checks. Overlapping windows are sensitivity checks.

The optional `plot` extra provides Matplotlib for `plot_repeat_check`. Core evaluation uses NumPy and SciPy. The C++ conditional-logit likelihood backend and R-based research examples are optional.

From a source checkout, run `uv run --frozen pytest -q`. A released wheel is installable with standard `pip install`.
