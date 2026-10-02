# Latest RM driver ratings

## 2026 · After Round 14 · Spanish Grand Prix

**2026 第14站（西班牙站）排位赛后的车手评分 · RM · 均值 6.5／标准差 2 展示刻度**

**22 drivers · RM (default from 0.3.0) · sorted by unrounded composite rating**

[Interactive chart · English](https://sixian-li.github.io/f1-quali/en.html) ·
[中文](https://sixian-li.github.io/f1-quali/) · [JSON snapshot](2026-round-14-rm.json) ·
[RM method](../docs/rm.md) · [Preserved m1r1 table](m1r1.md) · [Project](../README.md)

Results include qualifying through **2026 R14, 12 September**, with an exclusive
information cutoff of **25 September 2026, 11:00 UTC**. The roster and team names
are those of R14. This fixed snapshot matches the chart's current diamonds;
smoothing the chart changes its displayed curve, not this unsmoothed table.

| Rank | Driver | Team at R14 | **Mapped rating** | Original RM / 100 | Teammate ability / 100 | Comparisons |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Max Verstappen | Red Bull | **10.94** | 96.76 | 89.84 | 236 |
| 2 | Lando Norris | McLaren | **9.58** | 90.93 | 75.00 | 165 |
| 3 | Charles Leclerc | Ferrari | **9.07** | 88.38 | 72.76 | 186 |
| 4 | George Russell | Mercedes | **8.81** | 86.64 | 66.12 | 164 |
| 5 | Kimi Antonelli | Mercedes | **8.50** | 84.34 | 66.33 | 38 |
| 6 | Oscar Piastri | McLaren | **8.34** | 82.96 | 66.72 | 84 |
| 7 | Lewis Hamilton | Ferrari | **8.27** | 82.28 | 64.56 | 337 |
| 8 | Arvid Lindblad | Racing Bulls | **7.77** | 76.92 | 75.13 | 14 |
| 9 | Oliver Bearman | Haas | **7.76** | 76.79 | 76.79 | 38 |
| 10 | Fernando Alonso | Aston Martin | **7.51** | 74.52 | 74.52 | 291 |
| 11 | Carlos Sainz Jr. | Williams | **7.50** | 74.43 | 74.43 | 240 |
| 12 | Gabriel Bortoleto | Audi | **7.16** | 70.36 | 70.36 | 37 |
| 13 | Pierre Gasly | Alpine | **6.87** | 66.09 | 64.68 | 183 |
| 14 | Nico Hülkenberg | Audi | **6.73** | 64.68 | 64.68 | 263 |
| 15 | Esteban Ocon | Haas | **6.21** | 59.25 | 59.25 | 188 |
| 16 | Liam Lawson | Red Bull | **6.18** | 58.99 | 56.43 | 48 |
| 17 | Sergio Pérez | Cadillac | **5.79** | 54.66 | 54.66 | 294 |
| 18 | Alexander Albon | Williams | **5.74** | 53.82 | 53.82 | 130 |
| 19 | Valtteri Bottas | Cadillac | **5.36** | 49.22 | 49.22 | 256 |
| 20 | Yuki Tsunoda | Racing Bulls | **4.72** | 40.51 | 40.51 | 112 |
| 21 | Lance Stroll | Aston Martin | **4.33** | 36.02 | 36.02 | 199 |
| 22 | Franco Colapinto | Alpine | **4.05** | 32.67 | 32.67 | 41 |

### Reading the scores

- **Mapped rating** uses the frozen percentile reference of 7,226 pre-qualifying
  RM scores from 2010–2026 R14. The reference median maps to **6.5**, with target
  **SD 2**. This is not a score out of 10: values above 10 or below 0 are allowed.
  The 22 current endpoints are not part of the reference fit. Mapping is for
  display only and never enters historical prediction features or training.
- **Original RM** is the composite on its 1–100 scale: teammate ability plus
  fixed 0.6 times a shrunk summary of each driver's own final qualifying positions,
  with shared corrections learned on earlier years. It includes car and seat effects.
- **Teammate ability** retains the same 1–100 scale without the final-position
  contribution. It combines long-term and annual form plus the shared correction.
  Below-neutral recent results produce no bonus, so the original columns can match.
- **Faster responses:** less-experienced drivers have greater annual-state variance
  in both directions. Final-position means have a three-month half-life; reliability
  has a one-year half-life and prior mass 20. No returning-driver decay is included.
- **Comparisons** counts usable historical teammate comparisons since 2010.
  Pre-2010 participations affect career experience only. Counts are not confidence
  intervals. These scores are not win probabilities or championship points.

Circuit effects are kept separate from this general leaderboard. `f1-quali ratings`
also exports the driver-by-layout table. Sorting uses full precision; the table
shows two decimals. [Chart controls and mapping](../docs/rating_explorer.md).

### Snapshot and reproduction

The 2026 annual correction rules use 2010–2025 training data. Original rating
values exactly match the selected RM research case in the checked environment;
see [RM verification](../benchmarks/rm_verification.json). Reproduce with
`f1-quali ratings` and `scripts/export_leaderboard.py` after the documented RM
workflow. The JSON retains full-precision values, stable identities, source hashes,
cutoff and display-mapping identity.

The preserved [m1r1 snapshot](2026-round-14-m1r1.json) and earlier
[v6 snapshot](2026-round-14.json) remain available for comparison. They use different
rating methods; the v6 snapshot also has an earlier cutoff of 15 September 2026.

Historical identities and evidence derive from [F1DB](https://github.com/f1db/f1db)
`v2026.14.0` (CC BY 4.0), versioned schedules and reviewed official sources. Ratings
are project-derived estimates. See [third-party notices](../THIRD_PARTY_NOTICES.md)
and [data reproduction limits](../docs/data_and_reproduction.md#three-reproducibility-levels).
