# RM: rookie response and recent qualifying results

RM is the default in version **0.3.0**. It preserves m1r1's causal teammate
network and annual four-pool prediction pipeline, with two fixed changes:

- **R:** each driver's annual-state innovation variance is multiplied by
  `1 + 7 * 2**(-n / 20)`, where `n` counts main-qualifying appearances available
  strictly before January 1 of that year. This allows larger changes in both
  directions; it adds no positive drift or points for being young. The lifetime
  and circuit-layout priors are unchanged.
- **M:** individually admitted final qualifying positions have a **three-month
  mean half-life**, while reliability retains its **one-year half-life and prior
  mass 20**. There is no permanent achievement floor; beta stays fixed at 0.6.

There is no extra current-pair tracker, returning-driver decay, or driver-specific
adjustment. The same five bounded correction rules and four prediction heads
are refitted each year on earlier years. Both the original m1r1 and v6 remain
available through `--method m1r1` and `--method v6` on `prepare` and `demo`.
Later commands follow the method in their input manifests.

## Experience and the historical boundary

`full-era` now stores `experience.parquet` and `experience.json` from the pinned
F1DB release. Main-Q participations are deduplicated per driver/event and explicit
`DNS` and `NOT_PARTICIPATED` rows are excluded. Race date plus 36 hours supplies
a conservative reconstructed availability bound, not an original publication time.
Sprint qualifying and practice substitutes do not count as a regular driver's Q.

Pre-2010 appearances affect **experience counts only**. Rating observations and
prediction training still start in 2010. Unknown career coverage raises an error
instead of classifying an established driver as a rookie. Profiles are frozen at
each year boundary and retain their counts and variance multipliers in the
prepared artifact. Within-year results do not retrospectively alter that profile.
Across a missing-season gap, the AR forecast sums each calendar year's scaled
innovation variance; it does not reuse just the return year's multiplier.

Datasets prepared by older releases may lack career history. Re-run `full-era`
with the verified cache into a **new output directory** before preparing RM.
The existing m1r1/v6 inputs remain usable for those methods. Missing source bytes
remain a known limitation; see [data and reproduction](data_and_reproduction.md).

## Annual fitting and compatibility

For year Y, the predictor uses complete events from 2010 through Y−1, excluding
the first five rounds from fitting. RM rebuilds causal network snapshots and
teammate innovations under its rookie prior, then applies its achievement memory.
To reproduce the chosen research experiment, feature scaling and the initial Top3
fit use the original control anchors. The final shared corrections and all four
heads are fitted on the RM inputs. Simply changing an m1r1 memory constant after
training would not reproduce this method.

Artifacts are sealed with file hashes and a method/configuration identity.
Mixing an RM prepared history with an m1r1 annual model is rejected. The
implementation needs no research directory, registry, experiment runner or
pretrained private model. The bundled demo is synthetic and runs offline.

## Scores and the published chart

Predictions continue to consume **original RM values**. `ratings` exports
`ability_score` on the original 1–100 scale and an additional `mapped_score`
matching the chart's fixed normal-percentile scale: **mean/median 6.5, SD 2**.
The packaged reference is `rm-normal-95a8937f20a9e29a`, fitted to 7,226
pre-qualifying snapshots from 2010 through 2026 R14. The 22 current endpoints
are excluded from fitting it. Ties share a midrank; interpolation and outer-slope
extrapolation have no 0 or 10 cap.

This mapping is retrospective display calibration. It never enters historical
features, annual training, or forecast probabilities. The original score remains
available for audit, and sorting is unchanged. The standalone bilingual chart
applies both user-selected trailing averages to the original score **before**
mapping. See [chart controls and sources](rating_explorer.md).

## Evidence and limitations

The selected RM corresponds to the R+M case of the fixed 2010–2026 experiment.
Its improved response to rookie results is the reason for this product choice;
it is not a claim of consistent predictive superiority. The development comparison
used 2017–2022; 2023–2026 are already-seen diagnostics. In 2026, all eight research
variants still selected 12 of 24 Top3 members, and RM's Top3 probability error
was slightly worse than m1r1's. Larger rookie variance also increases early swings.

Current RM ratings and benchmarks have the same data boundary as the chart:
results through **2026 R14**, with the latest rating cutoff at
**2026-09-25 11:00 UTC**. No later results were added in this port. For numerical
port verification and event-group coverage, consult the repository's benchmark
records; those checks are distinct from claims about future prediction quality.
