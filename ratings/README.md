# Latest driver ratings

## 2026 · After Round 14 · Spanish Grand Prix

**2026 第14站（西班牙站）排位赛后的车手评分**

**22 drivers · m1r1 · sorted from highest to lowest composite rating**

[Back to the project](../README.md#driver-ratings) · [Download the score snapshot](2026-round-14-m1r1.json) · [Rating method](../docs/method.md#driver-ratings)

This is the latest published rating snapshot in this repository, computed with the
selected **m1r1** method. It includes qualifying results through **Round 14,
12 September 2026**, with an information cutoff of **25 September 2026, 11:00 UTC**
(before any Round 15 session result). The driver roster and team labels are those
of Round 14. Subsequent rounds are not included.

| Rank | Driver | Team at R14 | **Rating / 100** | Teammate ability / 100 | Comparisons |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Max Verstappen | Red Bull | **96.73** | 88.45 | 236 |
| 2 | Charles Leclerc | Ferrari | **89.26** | 75.67 | 186 |
| 3 | Lando Norris | McLaren | **88.71** | 72.27 | 165 |
| 4 | George Russell | Mercedes | **85.27** | 68.00 | 164 |
| 5 | Oscar Piastri | McLaren | **80.06** | 62.35 | 84 |
| 6 | Lewis Hamilton | Ferrari | **79.55** | 67.66 | 337 |
| 7 | Carlos Sainz Jr. | Williams | **77.13** | 76.39 | 240 |
| 8 | Kimi Antonelli | Mercedes | **76.76** | 61.90 | 38 |
| 9 | Oliver Bearman | Haas | **73.47** | 73.47 | 38 |
| 10 | Fernando Alonso | Aston Martin | **73.21** | 73.21 | 291 |
| 11 | Gabriel Bortoleto | Audi | **70.19** | 70.19 | 37 |
| 12 | Pierre Gasly | Alpine | **69.67** | 69.67 | 183 |
| 13 | Arvid Lindblad | Racing Bulls | **67.75** | 66.20 | 14 |
| 14 | Nico Hülkenberg | Audi | **66.41** | 66.41 | 263 |
| 15 | Esteban Ocon | Haas | **57.40** | 57.40 | 188 |
| 16 | Sergio Pérez | Cadillac | **57.05** | 57.05 | 294 |
| 17 | Alexander Albon | Williams | **56.17** | 56.17 | 130 |
| 18 | Valtteri Bottas | Cadillac | **54.83** | 54.83 | 256 |
| 19 | Liam Lawson | Red Bull | **46.87** | 46.87 | 48 |
| 20 | Yuki Tsunoda | Racing Bulls | **39.29** | 39.29 | 112 |
| 21 | Franco Colapinto | Alpine | **37.87** | 37.87 | 41 |
| 22 | Lance Stroll | Aston Martin | **33.64** | 33.64 | 199 |

### Reading the table

- **Rating** is the composite qualifying score: teammate ability plus 0.6 × a shrunk
  summary of the driver's own recent final qualifying positions, with small
  learned shared corrections. It includes some car and seat effects.
- **Teammate ability** is the teammate-based component (long-term ability plus
  season-level form, with the same small correction). The two scores use the same
  1–100 scale; their difference reflects the final-position contribution. Drivers
  whose recent positions are at or below the neutral middle receive no bonus, so
  both columns are equal.
- **Final positions fade within about a year.** Each driver's own valid positions
  count individually; their average and reliability both halve every year and
  shrink towards neutral. Older teammate comparisons keep a positive weight.
- **Comparisons** counts usable historical teammate comparisons admitted to the
  rating model, with evidence starting in 2010. Older observations receive different
  weights. Counts are not an uncertainty interval, and new drivers have less evidence.
- Scores are rating scales, not win probabilities or championship points. This is a
  general qualifying leaderboard; circuit-specific effects are kept separate
  (`f1-quali ratings` also writes the driver-by-layout table).
  Ordering uses the unrounded composite score; the page shows two decimals.

### Snapshot and sources

These are **post-R14 ratings** from the 2026 annual m1r1 model (correction rules
trained on 2010–2025). The published values equal the locked research selection
`m1r1_full_era_de17567786cf` exactly; see the
[verification record](../benchmarks/m1r1_verification.json). Regenerate with
`f1-quali ratings` and `scripts/export_leaderboard.py`.

The [JSON snapshot](2026-round-14-m1r1.json) retains full-precision values, stable
identities, cutoff and source hashes. This is a rating report, not an accuracy
evaluation. The earlier v6 snapshot for the same round remains available as
[2026-round-14.json](2026-round-14.json) (cutoff 15 September 2026); its scale is
not directly comparable because v6 weighted final positions differently.

Driver and team identities and historical evidence derive from
[F1DB](https://github.com/f1db/f1db) `v2026.14.0` (CC BY 4.0), versioned schedules and
reviewed official sources. The scores are project-derived estimates. See
[third-party notices](../THIRD_PARTY_NOTICES.md) and the
[data reproduction limits](../docs/data_and_reproduction.md#three-reproducibility-levels).
