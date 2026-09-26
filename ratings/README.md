# Latest driver ratings

## 2026 · After Round 14 · Spanish Grand Prix

**2026 第14站（西班牙站）排位赛后的车手评分**

**22 drivers · v6 · sorted from highest to lowest composite rating**

[Back to the project](../README.md#driver-ratings) · [Download the score snapshot](2026-round-14.json) · [Rating method](../docs/method.md#independent-driver-ratings)

This is the latest published rating snapshot in this repository. It includes qualifying
results through **Round 14, 12 September 2026**, with an information cutoff of
**15 September 2026, 00:00 UTC**. The driver roster and team labels are those of Round 14.
Subsequent rounds are not included. The snapshot was published on **26 September 2026**.

| Rank | Driver | Team at R14 | **Rating / 100** | Teammate ability / 100 | Comparisons |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Max Verstappen | Red Bull | **92.85** | 88.33 | 236 |
| 2 | Charles Leclerc | Ferrari | **81.05** | 75.47 | 186 |
| 3 | Lando Norris | McLaren | **78.39** | 71.90 | 165 |
| 4 | Carlos Sainz Jr. | Williams | **77.91** | 75.90 | 240 |
| 5 | Lewis Hamilton | Ferrari | **75.45** | 67.10 | 337 |
| 6 | Fernando Alonso | Aston Martin | **73.32** | 72.83 | 291 |
| 7 | Oliver Bearman | Haas | **73.00** | 73.00 | 38 |
| 8 | George Russell | Mercedes | **72.74** | 68.08 | 164 |
| 9 | Gabriel Bortoleto | Audi | **69.63** | 69.63 | 37 |
| 10 | Pierre Gasly | Alpine | **69.42** | 69.42 | 183 |
| 11 | Oscar Piastri | McLaren | **68.14** | 61.90 | 84 |
| 12 | Arvid Lindblad | Racing Bulls | **66.00** | 65.41 | 14 |
| 13 | Nico Hülkenberg | Audi | **65.91** | 65.91 | 263 |
| 14 | Kimi Antonelli | Mercedes | **65.66** | 60.89 | 38 |
| 15 | Sergio Pérez | Cadillac | **56.97** | 56.85 | 294 |
| 16 | Esteban Ocon | Haas | **56.87** | 56.87 | 188 |
| 17 | Alexander Albon | Williams | **55.97** | 55.97 | 130 |
| 18 | Valtteri Bottas | Cadillac | **55.20** | 54.33 | 256 |
| 19 | Liam Lawson | Red Bull | **46.40** | 46.40 | 48 |
| 20 | Yuki Tsunoda | Racing Bulls | **39.09** | 39.09 | 112 |
| 21 | Franco Colapinto | Alpine | **37.46** | 37.46 | 41 |
| 22 | Lance Stroll | Aston Martin | **32.76** | 32.76 | 199 |

### Reading the table

- **Rating** is the composite qualifying score used by v6: teammate ability plus a
  bounded contribution from historical final qualifying performance. It includes
  some car and seat effects.
- **Teammate ability** is the separate teammate-based component, combining long-term
  ability and season-level form. The two displayed scores use the same 1–100 scale;
  their difference reflects the achievement contribution after the display transform.
- **Comparisons** counts usable historical teammate comparisons admitted to the
  rating model, with evidence starting in 2010. Older observations receive different
  weights. Counts are not an uncertainty interval, and new drivers have less evidence.
- Scores are rating scales, not win probabilities or championship points. This is a
  general qualifying leaderboard; circuit-specific effects are kept separate.
  Ordering uses the unrounded composite score; the page shows two decimals.

### Snapshot and sources

These are **post-R14 ratings**: Round 14 evidence is included. The sealed snapshot
was also used as the rating input before Round 15, whose results are excluded.
The 22 displayed scores were checked against an independent refit using the
standalone v6 implementation; the maximum score difference was **0**.

The [JSON snapshot](2026-round-14.json) retains full-precision values, stable identities,
cutoff and source hashes. New publications will update this page while retaining
the dated JSON snapshots. This is a rating report, not an accuracy evaluation.

Driver and team identities and historical evidence derive from
[F1DB](https://github.com/f1db/f1db) `v2026.14.0` (CC BY 4.0), versioned schedules and
reviewed official sources. The scores are project-derived estimates. See
[third-party notices](../THIRD_PARTY_NOTICES.md) and the
[data reproduction limits](../docs/data_and_reproduction.md#three-reproducibility-levels).
