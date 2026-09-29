# Data and reproduction

## Three reproducibility levels

1. **Functional reproduction:** the bundled synthetic generator, offline tests and
   expected demo output need no Formula 1 data.
2. **Fixed historical reproduction:** obtain the pinned external source bytes,
   rebuild canonical tables, prepare eventwise ratings/features, and fit the recorded
   annual models. The complete m1r1 workflow was checked against the author's
   unpublished research implementation (lock `m1r1_full_era_de17567786cf`) using
   verified cached raw files: all dataset tables, prepared inputs, 17 annual models,
   7,226 historical predictions and the published ratings matched exactly
   ([record](../benchmarks/m1r1_verification.json)). The v6 workflow was checked the
   same way against lock `763e7f1871a6a6d3`.
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
- Official 2010–2015 qualifying pages and reviewed rules for the early seasons.
- Seven FIA rule and report documents (2013 Sporting Regulations and index,
  2014 US/Brazil/Abu Dhabi qualifying reports and a Brazil classification) that
  confirm the last common segment of 136 early teammate pairs.

The source recipes, URLs and hashes are packaged under `f1_quali/resources/`.
Raw ZIPs, HTML and PDFs are downloaded to the requested cache and are not bundled.
The selected model uses session summaries, so the experimental detailed FastF1 lap,
weather and tyre downloads are not required for reproduction.

## Full historical workflow

After the README installation steps, with the verified raw snapshots available
(a new download currently encounters the limitation above):

```sh
f1-quali data --cache data/raw --output data/canonical --offline
f1-quali full-era --data data/canonical --cache data/raw --output data/full_era --offline
f1-quali verify data/full_era
f1-quali prepare --data data/full_era --output runs/prepared
for year in $(seq 2010 2026); do
  f1-quali train --prepared runs/prepared --year "$year" --output "runs/models/$year"
  f1-quali evaluate --prepared runs/prepared --model "runs/models/$year" --output "runs/evaluations/$year"
done
python scripts/report_benchmarks.py --evaluations runs/evaluations --output runs/scorecard
f1-quali baseline --data data/canonical --years $(seq 2016 2026) --output runs/baselines
python scripts/report_m1r1_benchmarks.py --models runs/models --evaluations runs/evaluations \
  --baselines runs/baselines --output runs/scorecard
f1-quali ratings --prepared runs/prepared --model runs/models/2026 \
  --cutoff 2026-09-25T11:00:00Z --output runs/ratings
python scripts/export_leaderboard.py --ratings runs/ratings --data data/full_era \
  --cache data/raw --output runs/2026-round-14-m1r1.json
```

`--offline` reads only the existing verified cache and never downloads; omit it to
attempt a fresh download. No experiments or original course directory are required.
`full-era` adds 115 reconstructed 2010–2015 events to the 228 canonical events
(343 events, 7,226 entrant rows, 20,332 practice rows) and stores 1,298 early teammate
rows (1,249 outcome-usable, 1,164 with admitted pace, including the 136 reviewed
additions). `prepare` refits the teammate network before every event (a few minutes
single-threaded); each annual fit takes seconds.

`ratings` refits the network at the cutoff for the roster of the latest event with
results before it, applies the annual model's correction rules and writes the general
leaderboard plus a driver-by-layout table. Use the model of the cutoff's season; a
year-end table uses a cutoff such as `2025-12-31T23:59:59.999999999Z` with
`runs/models/2025`.

Outputs are checksummed and refuse conflicting overwrites. Use a new output
directory for changed data/configuration. `model.json` stores columns, scale,
coefficients, correction rules and training traces; its surrounding manifest records
integrity. Model loading needs neither the research code base nor a particular CPU
architecture. Dependencies are pinned, but other platforms still require actual
numerical validation.

### Previous method v6

```sh
f1-quali prepare --method v6 --data data/canonical --output runs/v6/prepared
for year in $(seq 2016 2026); do
  f1-quali train --prepared runs/v6/prepared --year "$year" --output "runs/v6/models/$year"
  f1-quali evaluate --prepared runs/v6/prepared --model "runs/v6/models/$year" \
    --output "runs/v6/evaluations/$year"
done
```

v6 uses the canonical dataset directly: 228 eventwise rating fits, with 2010–2015
supplying only rating warmup (1,298 teammate rows, 1,028 with admitted pace, and
1,655 complete-classification achievement rows).

## Canonical input contract and new events

`f1_quali.data.portable.load_dataset` returns a `Dataset` and optional warmup tables;
`f1_quali.data.full_era.load_full_era` returns a 2010+ `Dataset` and its early teammate
table. A dataset consists of `events`, `entries`, `qualifying`, `practice`,
`sessions`, `drivers`, and `calendar` Parquet tables, plus provenance and a manifest.
The synthetic generator provides a small executable example of these schemas.

For a new event, provide its reviewed layout/session identity, explicit qualifying
start, complete pre-cutoff roster, and any pre-cutoff FP results. Keep qualifying
results absent until they are available; set the event's result status accordingly.
Register a new driver identity even when their rating history is empty. Preserve
practice substitutes as distinct drivers. Append these rows to the full-era tables and
save a new dataset with `save_full_era` (or `save_dataset` for v6); do not edit a
sealed dataset in place.

```sh
f1-quali predict --data data/full-era-with-new-event --prepared runs/prepared \
  --model runs/models/2026 --event-id EVENT_ID --cutoff YYYY-MM-DDTHH:MM:SSZ \
  --output runs/new-forecast
```

m1r1 prediction reads the rating evidence and teammate innovations from the prepared
history. If the dataset contains any result available before the cutoff that the
prepared history lacks, the command refuses; re-run `prepare` on a dataset that
includes the new results first. `EVENT_ID` is the event's integer `event_id`; the
cutoff needs an explicit timezone. Inputs must belong to the model year. The release
has reviewed rules and annual rating parameter records through 2026; a later year
requires a reviewed configuration release. The pinned data command is not a
live-updating schedule service. For v6, omit `--prepared` and pass a v6 model.

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

For m1r1, 2010–2015 is a retrospective reconstruction: actual session clocks and
original publication times are unknown and stay null, practice is admitted by
reviewed session order, results are available from race date + 36 hours, and rosters
are post-event actual participants. Actual session timestamps are never invented. In
the v6 warmup, early records instead use the race-date bound for age and the
following January 1 for usable availability. Conditions, deleted laps, disruptions and some historical stage rules
remain incompletely observed. Missingness and exclusions are preserved. More raw
rows do not imply complete event labels or reliable fine-grained pace comparisons.
