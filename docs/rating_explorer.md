# Interactive driver rating history

**[Open the chart](https://sixian-li.github.io/f1-quali/)** to compare any of the
84 drivers with available m1r1 snapshots from 2010–2026. Search by name or
abbreviation, select multiple drivers, filter years, zoom, choose raw or smoothed
curves, and export the selected viewport as CSV. The interface is in Chinese;
driver names and abbreviations are searchable in English.

The published snapshot contains **7,248 points**: 7,226 historical pre-qualifying
ratings across 343 events, plus 22 current ratings after **2026 Round 14**. The
current cutoff is **2026-09-25 11:00 UTC**. This is a fixed snapshot, not a live feed.
The scores use the selected [m1r1 method](method.md), with the same current values
as the [22-driver leaderboard](../ratings/README.md).

## Reading the chart

- Circles show the rating available before the corresponding qualifying session.
  The final diamonds include the Round 14 results. Historical points do not use
  future results; the last point of an old season is not its post-season rating.
- Missing events and seasons are left blank. Dotted lines join consecutive seasons
  when the driver was present at both boundary events; annual fitted rules may change.
- The optional smoothing applies two trailing three-point simple moving averages
  to each driver's observed snapshots. Short initial windows use available points;
  years do not reset the window and absences are not filled. Tooltips and the table
  retain raw values. Smoothing does not alter the model.
- Dates in 2010–2015 are historical ordering proxies, not verified actual qualifying
  timestamps. Pre-2010 evidence is excluded, so the early history starts from a
  neutral prior. Scores are not a definitive cross-era ranking or win probabilities.

## Sharing and publication

Share the chart URL directly. Alternatively, download `docs/index.html` and open
it locally in a browser. All chart data, JavaScript and styles are embedded; no
Python environment, account, external fonts or online chart service is required.
When redistributing the HTML, retain its source attribution and the accompanying
[Plotly license](PLOTLY-LICENSE.txt).

GitHub Pages publishes the `main` branch's `/docs` directory. `.nojekyll` keeps the
prebuilt HTML unchanged. Updates require a reviewed replacement of the snapshot,
its cutoff labels and this note, followed by a commit to `main`; the page does not
fetch races or retrain anything. Research archives, fitted model files and private
work logs are not included in this publication.

The source HTML was checked against the locked historical scores and current
leaderboard. It contains exactly the original embedded data, application code and
styles; this public copy adds project and license links in the footer. Source HTML
SHA-256: `f1191c29e1b9fc354febd66122a5e1e4f073a5fc02b59ad3273414edcb511d9c`.

Historical identities and racing facts come from F1DB `v2026.14.0`; ratings are
calculated by this project. See [third-party notices](../THIRD_PARTY_NOTICES.md)
for source attribution and licenses.
