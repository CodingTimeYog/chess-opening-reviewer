# Opening coverage: book agent, test split

Git `3626f62`, seed 0. Definitions are in `chess_trainer/eval/coverage.py`.

## Position coverage (non-fallback move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| all positions | 65.2% (13,313 / 20,427) | 61.7% (15,200 / 24,631) |
| strict positions (never seen in train) | 0.0% (0 / 7,114) | 0.0% (0 / 9,431) |

## Line coverage (at least 10 consecutive known own moves)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| agent on the repertoire side | 81.3% (421 / 518) | 75.8% (472 / 623) |
| mean depth, repertoire side | 12.90 | 11.72 |
| agent on the opponent side | 76.8% (398 / 518) | 82.5% (517 / 627) |
| mean depth, opponent side | 12.40 | 12.22 |

## Prep coverage (repertoire side to move, prep tree has a repertoire move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| positions | 36.8% (3,761 / 10,223) | 37.4% (4,606 / 12,303) |

## Move quality: agreement with the master move (every held-out position)

| | Ruy Lopez, all | Ruy Lopez, strict | Sicilian Taimanov, all | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 20,427 | 7,114 | 24,631 | 9,431 |
| Top-1 agreement | 52.4% | 3.3% | 51.1% | 3.1% |
| Top-3 agreement | 65.0% | 9.6% | 60.8% | 9.0% |

Sanity checks over 45,058 positions: 0 policies containing an illegal move, 0 policies that are empty or do not sum to 1, 0 illegal master moves.

## Move quality: Stockfish depth 14 (seeded sample of distinct positions, seed 0)

| | Ruy Lopez, seen | Ruy Lopez, strict | Sicilian Taimanov, seen | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 500 | 500 | 500 | 500 |
| out of book (fallback) | 0.0% | 100.0% | 0.0% | 100.0% |
| mean CPL, top move | 3.5 | 248.5 | 5.8 | 297.2 |
| median CPL, top move | 0.0 | 176.5 | 0.0 | 242.0 |
| blunder rate (CPL >= 100), top move | 0.0% | 62.8% | 0.0% | 72.4% |
| mean WDL expectation loss, top move | 0.006 | 0.325 | 0.015 | 0.359 |
| mean CPL, sampled | 4.3 | 247.6 | 7.2 | 282.1 |
| median CPL, sampled | 1.7 | 204.8 | 2.9 | 235.3 |
| blunder rate (CPL >= 100), sampled | 0.0% | 62.1% | 0.1% | 69.5% |
| mean WDL expectation loss, sampled | 0.007 | 0.326 | 0.018 | 0.350 |
| mean CPL, master | 4.2 | 9.8 | 7.9 | 13.8 |
| median CPL, master | 0.0 | 0.0 | 0.0 | 0.0 |
| blunder rate (CPL >= 100), master | 0.0% | 0.4% | 0.2% | 1.8% |
| mean WDL expectation loss, master | 0.008 | 0.019 | 0.020 | 0.032 |

`top move`: the agent's most probable move (a uniform fallback is one seeded random legal move). `sampled`: expectation over the agent's whole policy. `master`: the move actually played, for reference. For `sampled`, the median is over per-position expected CPL.
