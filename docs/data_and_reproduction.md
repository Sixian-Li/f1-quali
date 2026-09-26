# Data and reproduction

## Three reproducibility levels

1. **Functional reproduction:** the bundled synthetic generator, offline tests and
   expected demo output need no Formula 1 data.
2. **Fixed historical reproduction:** obtain the pinned external source bytes,
   rebuild canonical tables, prepare eventwise ratings/features, and fit the recorded
   annual models. The complete workflow was checked against the author's unpublished
   research implementation of v6 (research lock ID `763e7f1871a6a6d3`) using verified
   cached raw files; all nine core/warmup tables matched exactly.
3. **Fresh remote acquisition:** download those same pinned versions again. Website
   availability is external to this repository. A fresh-cache attempt on 2026-09-26
   failed at the first required Formula 1 evidence article with a byte-checksum
   mismatch. No downloaded replacement was admitted and the remaining sources were
   not exhaustively checked. See [the recorded check](../benchmarks/source_acquisition.json).
   Exact historical reproduction therefore requires the original verified snapshots,
   which are not distributed with this repository.
   A changed hash or inaccessible source is never permission to silently use a newer
   source. Reviewing replacement snapshots would create a new data version, outside
   this behavior-preserving refactor.

## Fixed sources

- F1DB `v2026.14.0`, ZIP SHA-256
  `8a92b898bc237bd3b0a86fe5665ebe0e8f8328f126e2422ca9946d175d66dfb4`.
- f1schedule commit `5c807120ae154922fd0075c3583bc2bbf78b80ff`, individual yearly hashes.
- Explicit session, entrant and classification decisions, each with a source URL
  and evidence key. They are interpretations recorded by this project, not new
  original-publication timestamps.
- Official 2010–2015 qualifying pages and reviewed rules for rating warmup.

The source recipes, URLs and hashes are packaged under `f1_quali/resources/`.
Raw ZIPs, HTML and PDFs are downloaded to the requested cache and are not bundled.
The selected model uses session summaries, so the experimental detailed FastF1 lap,
weather and tyre downloads are not required for v6 reproduction.

## Full historical workflow

After the README installation steps, with the verified raw snapshots available
(a new download currently encounters the limitation above):

```sh
f1-quali data --cache data/raw --output data/canonical --offline
f1-quali verify data/canonical
f1-quali prepare --data data/canonical --output runs/prepared
for year in 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
  f1-quali train --prepared runs/prepared --year "$year" --output "runs/models/$year"
  f1-quali evaluate --prepared runs/prepared --model "runs/models/$year" --output "runs/evaluations/$year"
done
python scripts/report_benchmarks.py --evaluations runs/evaluations --output runs/scorecard
f1-quali baseline --data data/canonical --years 2024 2026 --output runs/baselines
```

`--offline` reads only the existing verified cache and never downloads; omit it to
attempt a fresh download. No experiments or
original course directory are required. A full run is larger than the demo: 228
eventwise rating fits, 4,629 entrant rows, 12,614 practice rows and 11 annual fits.
2010–2015 contributes 1,298 stored teammate rows (1,249 outcome-usable, 1,028 with
admitted pace) and 1,655 complete-classification achievement rows. No early-year
row becomes a predictor training example.

Outputs are checksummed and refuse conflicting overwrites. Use a new output
directory for changed data/configuration. `model.json` stores columns, scale,
coefficients and training traces; its surrounding manifest records integrity.
Model loading needs neither the research code base nor a particular CPU architecture.
Dependencies are pinned, but other platforms still require actual numerical validation.

## Canonical input contract and new events

`f1_quali.data.portable.load_dataset` returns a `Dataset` and optional early rating
tables. A dataset consists of `events`, `entries`, `qualifying`, `practice`, `sessions`,
`drivers`, and `calendar` Parquet tables, plus provenance and a manifest. The
synthetic generator provides a small executable example of these schemas.

For a new event, provide its reviewed layout/session identity, explicit qualifying
start, complete pre-cutoff roster, and any pre-cutoff FP results. Keep qualifying
results absent until they are available; set the event's result status accordingly.
Register a new driver identity even when their rating history is empty. Preserve
practice substitutes as distinct drivers. Save a new dataset with `save_dataset`;
do not edit a sealed dataset in place.

```sh
f1-quali predict --data data/canonical-with-new-event --model runs/models/2026 \
  --event-id EVENT_ID --cutoff YYYY-MM-DDTHH:MM:SSZ --output runs/new-forecast
```

The dataset holds the full canonical history with the new event appended, not the new
event alone. `EVENT_ID` is that event's integer `event_id`; the cutoff needs an explicit
timezone. Inputs must belong to
the model year. The release has reviewed rules and annual rating parameter records
through 2026; a later year requires a reviewed configuration release. The pinned
data command is not a live-updating schedule service.

Prediction outputs retain driver/constructor identity, missing-practice flags,
rating provenance, raw/coordinated pool probabilities and the nested hard selections.
`forecast.json` records both the requested cutoff and actual generation time.
A historical reconstruction generated after qualifying is not a prospective forecast.

## Availability limitations

Most historical entrant memberships and source revisions were collected
retrospectively. Original publication times are unknown. Null entrant publication
timestamps remain null with an explicit retrospective membership basis; they are
not invented timestamps. Session availability uses documented reconstruction rules
and decision buffers. Scheduled times are not measured session start/end times.

Early records use a race-date bound for age and the following January 1 for usable
availability. Conditions, deleted laps, disruptions and some historical stage rules
remain incompletely observed. Missingness and exclusions are preserved. More raw
rows do not imply complete event labels or reliable fine-grained pace comparisons.
