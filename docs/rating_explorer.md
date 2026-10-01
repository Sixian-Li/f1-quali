# Interactive RM research rating history

**[English](https://sixian-li.github.io/f1-quali/en.html) · [中文版](https://sixian-li.github.io/f1-quali/)** — compare any of the
84 drivers with available **RM** snapshots from 2010–2026. Search by name or
abbreviation, select multiple drivers, filter years, zoom, adjust two smoothing
windows and export the selected viewport as CSV. Both languages include the
controls, hover labels, help text and source notes. Use **中文 / English** in the
header to switch; selected drivers, years, zoom and moving-average windows carry
over when browser storage is available. Names, abbreviations and supported
Chinese aliases are searchable in either version.

The published snapshot contains **7,248 points**: 7,226 historical pre-qualifying
ratings across 343 events, plus 22 current ratings after **2026 Round 14**. The
current cutoff is **2026-09-25 11:00 UTC**. This is a fixed snapshot, not a live feed.

## Display scale: mean 6.5, standard deviation 2

The chart uses a frozen historical-percentile mapping to a normal distribution
with **mean and median 6.5, standard deviation 2**. It is not a score out of 10:
there is no upper or lower cap. The reference sample spans approximately
**−1.12 to 14.12**; the top 1% threshold is approximately **11.15**.

The reference pools the **7,226 pre-qualifying RM ratings** from 2010 Round 1
through 2026 Round 14, with equal weight per driver-event. The 22 extra current
endpoints are excluded from fitting the scale. For a raw score `x`, tied scores
share the midrank percentile:

```text
p = (number below x + 0.5 × number equal to x) / 7,226
mapped score = 6.5 + 2 × inverse_standard_normal_CDF(p)
```

The 7,139 unique raw scores define fixed mapping knots. Values between knots use
linear interpolation in raw-score/mapped-score space; values outside the
reference continue the nearest segment's slope. Selecting drivers, years or
smoothing windows does not refit this reference. Ties are preserved, so the
reference's actual population standard deviation is approximately 1.999816.

This transformation preserves ordering but changes score differences. It is a
retrospective display calibration using the whole reference period, not a feature
that was available to historical forecasts. The original cutoff-aware RM scores
remain embedded unchanged; the mapping never enters the prediction model.
Mapping ID: `rm-normal-95a8937f20a9e29a`.

## RM and the package's selected model

RM is a research variant of [m1r1](method.md):

- **R: more room for rookie changes.** The season-level innovation variance is
  multiplied by `1 + 7 * 2^(-n / 20)`, where `n` counts prior main-qualifying
  appearances at the start of the year. This permits larger moves in either
  direction; it does not automatically award points to rookies.
- **M: more weight on recent final positions.** The achievement mean uses a
  three-month half-life. Its reliability still uses a one-year half-life and prior
  mass 20; the achievement weight remains 0.6. Driver admission and the bounded
  shared correction rules follow the same structure as m1r1, with annual rules
  refitted for this variant on earlier years.

This publication updates the **chart snapshot only**. The Python package's
selected method, CLI, [22-driver leaderboard](../ratings/README.md), and published
prediction benchmarks remain **m1r1**. They therefore need not match RM scores.
The package does not provide an RM training command; the page displays saved RM
outputs from the research implementation.

RM remains a research candidate for discussion. The comparison used 2017–2022 for
validation and 2023–2026 as already-seen diagnostics, not an untouched test set.
Higher responsiveness increases early-career volatility, and prediction metrics
were mixed; 2026 Top 3 selections did not improve in the research comparison.
A smoother-looking curve is not evidence of better forecasts.

## Reading and controlling the chart

- The initial view shows the usual eight drivers in 2025–2026 with unsmoothed
  mapped scores.
  Select other drivers or years, or use **ANT / RUS · 2026** for that pairing.
- Circles show mapped pre-qualifying rating snapshots. The final diamonds include
  the Round 14 results. The underlying historical RM ratings do not use future
  results; their display scale uses the full-period reference described above.
  The last point of an old season is not its post-season rating.
- Missing events and seasons are left blank. Dotted lines join consecutive seasons
  when the driver was present at both boundary events; annual fitted rules may change.
- Select **Two-pass SMA / 两次均线（可调）** to set the first and second trailing simple-moving-average
  windows independently. The second pass averages the first pass's output.
  Defaults are **3 / 3**; **1** bypasses that pass. Larger windows are smoother and
  more delayed. Short initial windows use available points; years do not reset the
  window and absences are not filled. The latest endpoint also enters the average.
- Both averages operate on the **original RM scores**, and the result is then
  mapped to the new display scale. Smoothing uses each driver's full prior
  history before applying the viewport filter, so zooming does not restart it.
  A smoothed curve need not itself have mean 6.5 or standard deviation 2.
- The chart, driver list and table use mapped scores; the table always shows
  unsmoothed mapped scores. Hover also reports the original RM score. CSV retains
  mapped raw/smoothed scores, original RM raw/smoothed scores, the display mode,
  both windows, mapping parameters and version, and missing-teammate-evidence
  flags. Filtering and smoothing do not change the model.
- Dates in 2010–2015 are historical ordering proxies, not verified actual qualifying
  timestamps. Pre-2010 evidence is excluded, so the early history starts from a
  neutral prior. Scores are not a definitive cross-era ranking or win probabilities.

## Sharing, sources and publication

Share either language URL directly. Alternatively, download `docs/index.html`
(Chinese) or `docs/en.html` (English) and open it locally in a browser. Download
both into the same folder to use the language links offline. Each file embeds
all chart data, JavaScript and styles; no
Python environment, account, external fonts or online chart service is required.
The complete Plotly license and source attribution are embedded in the HTML and
must be retained when sharing it. The license is also in
[Plotly license](PLOTLY-LICENSE.txt).

GitHub Pages publishes the `main` branch's `/docs` directory. `.nojekyll` keeps the
prebuilt HTML unchanged. Updates require a reviewed replacement of the snapshot,
its cutoff labels and this note, followed by a commit to `main`; the page does not
fetch races or retrain anything. Research archives, fitted model files and private
work logs are not included in this publication.

All RM points retain exactly matching original scores, default 3/3 smoothing,
time and driver identities, and gap markers. Both language pages use the same
frozen mapping and apply it only to derived display values. The published pages
match the reviewed offline mean-6.5/SD-2 files except for the language-link
filenames. Original Chinese source HTML SHA-256:
`1147e43bfd57fbd7d0428f244710b295e26a22fa981dadce5602ddde4a0e3955`;
English source SHA-256:
`2992e39dcc2f7d880a724a5cdc9d08a1156910627a516aeaa5dd13299b89c887`.
The previous original-scale RM and m1r1 pages remain available in Git history.

The Chinese page is the source for shared markup, styles and application logic.
After an interface edit, update `scripts/rating_page_en.json` and run:

```sh
python scripts/build_rating_languages.py
python scripts/build_rating_languages.py --check
node scripts/test_rating_pages.mjs
```

The builder leaves the bundled Plotly library and rating JSON untouched and
fails on missing or stale translations. The offline JavaScript checks execute
both pages' actual application scripts with DOM and Plotly test doubles, check
the original RM records and frozen mapping against pinned hashes, and verify all
mapping knots, smoothing-before-mapping, uncapped axes, language switching, CSV
and zoom behavior. They do not replace browser rendering checks.

Historical identities and racing facts come from F1DB `v2026.14.0`; ratings are
calculated by this project. See [third-party notices](../THIRD_PARTY_NOTICES.md)
for source attribution and licenses.
