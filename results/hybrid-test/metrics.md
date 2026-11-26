## hybrid vs book, resnet-main on strict positions (test split, never seen in train)

Agreement and coverage use every strict position (Ruy Lopez 7,114, Sicilian Taimanov 9,431); the Stockfish rows use the same seeded sample for every agent.

| | Ruy Lopez: book | Ruy Lopez: resnet-main | Ruy Lopez: hybrid | Sicilian Taimanov: book | Sicilian Taimanov: resnet-main | Sicilian Taimanov: hybrid |
|---|---:|---:|---:|---:|---:|---:|
| position coverage (non-fallback) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| Top-1 agreement with master | 3.3% | 30.6% | 30.9% | 3.1% | 29.8% | 30.0% |
| Top-3 agreement with master | 9.6% | 54.0% | 54.1% | 9.0% | 52.8% | 52.7% |
| mean CPL, top move | 248.5 | 89.8 | 89.0 | 297.2 | 108.2 | 107.9 |
| median CPL, top move | 176.5 | 24.5 | 24.0 | 242.0 | 33.0 | 31.0 |
| blunder rate (CPL >= 100), top move | 62.8% | 22.0% | 21.8% | 72.4% | 27.8% | 27.8% |
| mean WDL expectation loss, top move | 0.325 | 0.138 | 0.136 | 0.359 | 0.153 | 0.152 |
| mean CPL, sampling from the policy | 247.6 | 114.9 | 113.7 | 282.1 | 132.1 | 131.1 |
| mean CPL of the master's move (reference) | 9.8 | 9.8 | 9.8 | 13.8 | 13.8 | 13.8 |

## hybrid vs book, resnet-main: all positions and the seen sample

Agreement is over every held-out position; the Stockfish rows are the sample of positions that also occur in train.

| | Ruy Lopez: book | Ruy Lopez: resnet-main | Ruy Lopez: hybrid | Sicilian Taimanov: book | Sicilian Taimanov: resnet-main | Sicilian Taimanov: hybrid |
|---|---:|---:|---:|---:|---:|---:|
| Top-1 agreement with master, all positions | 52.3% | 61.4% | 61.9% | 51.1% | 60.9% | 61.4% |
| Top-3 agreement with master, all positions | 65.1% | 81.3% | 80.6% | 60.8% | 78.6% | 77.6% |
| mean CPL, top move | 3.5 | 8.4 | 3.5 | 5.8 | 13.4 | 5.8 |
| median CPL, top move | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| blunder rate (CPL >= 100), top move | 0.0% | 1.4% | 0.0% | 0.0% | 2.0% | 0.0% |
| mean WDL expectation loss, top move | 0.006 | 0.012 | 0.006 | 0.015 | 0.025 | 0.015 |
| mean CPL, sampling from the policy | 4.4 | 18.9 | 4.4 | 7.2 | 25.1 | 7.2 |
| mean CPL of the master's move (reference) | 4.2 | 4.2 | 4.2 | 7.9 | 7.9 | 7.9 |

# Opening coverage: hybrid agent, test split

Git `a7eb14c`, seed 0. Definitions are in `chess_trainer/eval/coverage.py`.

## Position coverage (non-fallback move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| all positions | 100.0% (20,427 / 20,427) | 100.0% (24,631 / 24,631) |
| strict positions (never seen in train) | 100.0% (7,114 / 7,114) | 100.0% (9,431 / 9,431) |

## Line coverage (at least 10 consecutive known own moves)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| agent on the repertoire side | 81.7% (423 / 518) | 76.9% (479 / 623) |
| mean depth, repertoire side | 12.96 | 11.79 |
| agent on the opponent side | 77.0% (399 / 518) | 83.1% (521 / 627) |
| mean depth, opponent side | 12.45 | 12.26 |

## Prep coverage (repertoire side to move, prep tree has a repertoire move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| positions | 36.8% (3,761 / 10,223) | 37.4% (4,606 / 12,303) |

## Which tier answers (book, then prep repertoire, then ResNet)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| % book, all positions | 65.2% | 61.7% |
| % prep, all positions | 0.3% | 0.3% |
| % resnet, all positions | 34.5% | 38.0% |
| % book, strict positions | 0.0% | 0.0% |
| % prep, strict positions | 0.9% | 0.8% |
| % resnet, strict positions | 99.1% | 99.2% |

## Move quality: agreement with the master move (every held-out position)

| | Ruy Lopez, all | Ruy Lopez, strict | Sicilian Taimanov, all | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 20,427 | 7,114 | 24,631 | 9,431 |
| Top-1 agreement | 61.9% | 30.9% | 61.4% | 30.0% |
| Top-3 agreement | 80.6% | 54.1% | 77.6% | 52.7% |

Sanity checks over 45,058 positions: 0 policies containing an illegal move, 0 policies that are empty or do not sum to 1, 0 illegal master moves.

## Move quality: Stockfish depth 14 (seeded sample of distinct positions, seed 0)

| | Ruy Lopez, seen | Ruy Lopez, strict | Sicilian Taimanov, seen | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 500 | 500 | 500 | 500 |
| out of book (fallback) | 0.0% | 0.0% | 0.0% | 0.0% |
| mean CPL, top move | 3.5 | 89.0 | 5.8 | 107.9 |
| median CPL, top move | 0.0 | 24.0 | 0.0 | 31.0 |
| blunder rate (CPL >= 100), top move | 0.0% | 21.8% | 0.0% | 27.8% |
| mean WDL expectation loss, top move | 0.006 | 0.136 | 0.015 | 0.152 |
| mean CPL, sampled | 4.4 | 113.7 | 7.2 | 131.1 |
| median CPL, sampled | 1.8 | 69.7 | 3.0 | 88.4 |
| blunder rate (CPL >= 100), sampled | 0.0% | 28.7% | 0.1% | 34.2% |
| mean WDL expectation loss, sampled | 0.008 | 0.166 | 0.018 | 0.182 |
| mean CPL, master | 4.2 | 9.8 | 7.9 | 13.8 |
| median CPL, master | 0.0 | 0.0 | 0.0 | 0.0 |
| blunder rate (CPL >= 100), master | 0.0% | 0.4% | 0.2% | 1.8% |
| mean WDL expectation loss, master | 0.008 | 0.019 | 0.020 | 0.032 |

`top move`: the agent's most probable move (a uniform fallback is one seeded random legal move). `sampled`: expectation over the agent's whole policy. `master`: the move actually played, for reference. For `sampled`, the median is over per-position expected CPL.
