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

- The initial view shows the usual eight drivers in 2025–2026 with raw scores.
  Select other drivers or years, or use **ANT / RUS · 2026** for that pairing.
- Circles show the rating available before the corresponding qualifying session.
  The final diamonds include the Round 14 results. Historical points do not use
  future results; the last point of an old season is not its post-season rating.
- Missing events and seasons are left blank. Dotted lines join consecutive seasons
  when the driver was present at both boundary events; annual fitted rules may change.
- Select **Two-pass SMA / 两次均线（可调）** to set the first and second trailing simple-moving-average
  windows independently. The second pass averages the first pass's output.
  Defaults are **3 / 3**; **1** bypasses that pass. Larger windows are smoother and
  more delayed. Short initial windows use available points; years do not reset the
  window and absences are not filled. The latest endpoint also enters the average.
- Smoothing uses each driver's full prior history before applying the viewport
  filter, so zooming does not restart it. Tooltips and the table retain raw values.
  CSV includes raw, smoothed and displayed scores plus both window settings.
  Smoothing only affects display; it does not change ratings or the model.
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

All RM points were checked against the source comparison, with exactly matching
raw scores, default 3/3 smoothing, time and driver identities, and gap markers.
Both language pages preserve the standalone RM file's embedded rating data
exactly. Translations, language links, shared viewport preferences and responsive
header styling are presentation changes; the rating calculations and saved
values are unchanged. Original offline source HTML
SHA-256: `6dc9c03b70687aed8356a8c0755f06efad64c40f148e7db0fc5d59a50159d629`.
The prior m1r1 page remains available in Git history.

The Chinese page is the source for shared markup, styles and application logic.
After an interface edit, update `scripts/rating_page_en.json` and run:

```sh
python scripts/build_rating_languages.py
python scripts/build_rating_languages.py --check
node scripts/test_rating_pages.mjs
```

The builder leaves the bundled Plotly library and rating JSON untouched and
fails on missing or stale translations. The offline JavaScript checks execute
both pages' actual application scripts with DOM and Plotly test doubles, compare
their data with the original RM snapshot hash, and verify language switching,
smoothing, CSV and zoom behavior. They do not replace browser rendering checks.

Historical identities and racing facts come from F1DB `v2026.14.0`; ratings are
calculated by this project. See [third-party notices](../THIRD_PARTY_NOTICES.md)
for source attribution and licenses.
