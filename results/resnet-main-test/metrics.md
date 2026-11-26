## resnet-main vs book on strict positions (test split, never seen in train)

Agreement and coverage use every strict position (Ruy Lopez 7,114, Sicilian Taimanov 9,431); the Stockfish rows use the same seeded sample for every agent.

| | Ruy Lopez: book | Ruy Lopez: resnet-main | Sicilian Taimanov: book | Sicilian Taimanov: resnet-main |
|---|---:|---:|---:|---:|
| position coverage (non-fallback) | 0.0% | 100.0% | 0.0% | 100.0% |
| Top-1 agreement with master | 3.3% | 30.6% | 3.1% | 29.8% |
| Top-3 agreement with master | 9.6% | 54.0% | 9.0% | 52.8% |
| mean CPL, top move | 248.5 | 89.8 | 297.2 | 108.2 |
| median CPL, top move | 176.5 | 24.5 | 242.0 | 33.0 |
| blunder rate (CPL >= 100), top move | 62.8% | 22.0% | 72.4% | 27.8% |
| mean WDL expectation loss, top move | 0.325 | 0.138 | 0.359 | 0.153 |
| mean CPL, sampling from the policy | 247.6 | 114.9 | 282.1 | 132.1 |
| mean CPL of the master's move (reference) | 9.8 | 9.8 | 13.8 | 13.8 |

## resnet-main vs book: all positions and the seen sample

Agreement is over every held-out position; the Stockfish rows are the sample of positions that also occur in train.

| | Ruy Lopez: book | Ruy Lopez: resnet-main | Sicilian Taimanov: book | Sicilian Taimanov: resnet-main |
|---|---:|---:|---:|---:|
| Top-1 agreement with master, all positions | 52.3% | 61.4% | 51.1% | 60.9% |
| Top-3 agreement with master, all positions | 65.1% | 81.3% | 60.8% | 78.6% |
| mean CPL, top move | 3.5 | 8.4 | 5.8 | 13.4 |
| median CPL, top move | 0.0 | 0.0 | 0.0 | 0.0 |
| blunder rate (CPL >= 100), top move | 0.0% | 1.4% | 0.0% | 2.0% |
| mean WDL expectation loss, top move | 0.006 | 0.012 | 0.015 | 0.025 |
| mean CPL, sampling from the policy | 4.4 | 18.9 | 7.2 | 25.1 |
| mean CPL of the master's move (reference) | 4.2 | 4.2 | 7.9 | 7.9 |

# Opening coverage: resnet-main agent, test split

Git `a7eb14c`, seed 0. Definitions are in `chess_trainer/eval/coverage.py`.

## Position coverage (non-fallback move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| all positions | 100.0% (20,427 / 20,427) | 100.0% (24,631 / 24,631) |
| strict positions (never seen in train) | 100.0% (7,114 / 7,114) | 100.0% (9,431 / 9,431) |

## Line coverage (at least 10 consecutive known own moves)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| agent on the repertoire side | 77.8% (403 / 518) | 70.8% (441 / 623) |
| mean depth, repertoire side | 12.41 | 11.26 |
| agent on the opponent side | 73.8% (382 / 518) | 79.6% (499 / 627) |
| mean depth, opponent side | 11.92 | 11.82 |

## Prep coverage (repertoire side to move, prep tree has a repertoire move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| positions | 36.8% (3,761 / 10,223) | 37.4% (4,606 / 12,303) |

## Move quality: agreement with the master move (every held-out position)

| | Ruy Lopez, all | Ruy Lopez, strict | Sicilian Taimanov, all | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 20,427 | 7,114 | 24,631 | 9,431 |
| Top-1 agreement | 61.4% | 30.6% | 60.9% | 29.8% |
| Top-3 agreement | 81.3% | 54.0% | 78.6% | 52.8% |

Sanity checks over 45,058 positions: 0 policies containing an illegal move, 0 policies that are empty or do not sum to 1, 0 illegal master moves.

## Move quality: Stockfish depth 14 (seeded sample of distinct positions, seed 0)

| | Ruy Lopez, seen | Ruy Lopez, strict | Sicilian Taimanov, seen | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 500 | 500 | 500 | 500 |
| out of book (fallback) | 0.0% | 0.0% | 0.0% | 0.0% |
| mean CPL, top move | 8.4 | 89.8 | 13.4 | 108.2 |
| median CPL, top move | 0.0 | 24.5 | 0.0 | 33.0 |
| blunder rate (CPL >= 100), top move | 1.4% | 22.0% | 2.0% | 27.8% |
| mean WDL expectation loss, top move | 0.012 | 0.138 | 0.025 | 0.153 |
| mean CPL, sampled | 18.9 | 114.9 | 25.1 | 132.1 |
| median CPL, sampled | 7.7 | 70.8 | 10.1 | 89.3 |
| blunder rate (CPL >= 100), sampled | 3.8% | 29.1% | 5.2% | 34.4% |
| mean WDL expectation loss, sampled | 0.028 | 0.168 | 0.044 | 0.184 |
| mean CPL, master | 4.2 | 9.8 | 7.9 | 13.8 |
| median CPL, master | 0.0 | 0.0 | 0.0 | 0.0 |
| blunder rate (CPL >= 100), master | 0.0% | 0.4% | 0.2% | 1.8% |
| mean WDL expectation loss, master | 0.008 | 0.019 | 0.020 | 0.032 |

`top move`: the agent's most probable move (a uniform fallback is one seeded random legal move). `sampled`: expectation over the agent's whole policy. `master`: the move actually played, for reference. For `sampled`, the median is over per-position expected CPL.
