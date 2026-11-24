# Changelog

## 1.0.1

- Added exhaustive and seeded random forecast perturbations and a separate prefix-only preprocessing check.
- Added explicit empirical win, tie and mean-sign support fields to the paired Monte Carlo summary. Existing field names and the two-block default remain available.
- Added an end-to-end public cohort example and repaired paired starts in the R examples.

## 1.0.0 — R22 statistical-semantics correction

- Renamed the `reference_band_screen` argument from `equivalence_margin` to `practical_margin`.
- Replaced the misleading `equivalent_within_margin` result with `predictive_interval_within_margin`.
- Added `margin_screen_basis` so downstream users can see that the rule uses the 95% interval of simulated cohort-level discrepancies.
- Clarified in code and documentation that this is a descriptive predictive-distribution screen conditional on the fitted model and simulation design, not a TOST, not a population-mean confidence interval, and not a frequentist equivalence test.
- Renamed `adequacy_set` output groups to `inside_reference_band` and `outside_reference_band`; these are descriptive flags, not formal adequacy decisions.
- Updated the calibration and IGT analysis scripts and the archived JSON keys to use practical-margin terminology.
- This is a semantic API-breaking change. The previous 0.9.1 API must not be mixed with the 1.0.0 manuscript tables.

## 0.9.1

See the archived R21 release materials for the previous API and analysis snapshot.
