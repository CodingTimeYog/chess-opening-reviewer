## resnet-main vs book on strict positions (test split, never seen in train)

Agreement and coverage use every strict position (Ruy Lopez 7,114, Sicilian Taimanov 9,431); the Stockfish rows use the same seeded sample for both agents.

| | Ruy Lopez: book | Ruy Lopez: resnet-main | Sicilian Taimanov: book | Sicilian Taimanov: resnet-main |
|---|---:|---:|---:|---:|
| position coverage (non-fallback) | 0.0% | 100.0% | 0.0% | 100.0% |
| Top-1 agreement with master | 3.3% | 30.8% | 3.1% | 30.0% |
| Top-3 agreement with master | 9.6% | 54.0% | 9.0% | 52.7% |
| mean CPL, top move | 248.5 | 85.9 | 297.2 | 103.9 |
| median CPL, top move | 176.5 | 19.0 | 242.0 | 33.5 |
| blunder rate (CPL >= 100), top move | 62.8% | 21.8% | 72.4% | 26.2% |
| mean WDL expectation loss, top move | 0.325 | 0.133 | 0.359 | 0.149 |
| mean CPL, sampling from the policy | 247.6 | 128.8 | 282.1 | 152.1 |
| mean CPL of the master's move (reference) | 9.8 | 9.8 | 13.8 | 13.8 |

# Opening coverage: resnet-main agent, test split

Git `3e94cac`, seed 0. Definitions are in `chess_trainer/eval/coverage.py`.

## Position coverage (non-fallback move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| all positions | 100.0% (20,427 / 20,427) | 100.0% (24,631 / 24,631) |
| strict positions (never seen in train) | 100.0% (7,114 / 7,114) | 100.0% (9,431 / 9,431) |

## Line coverage (at least 10 consecutive known own moves)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| agent on the repertoire side | 73.8% (382 / 518) | 65.7% (409 / 623) |
| mean depth, repertoire side | 11.74 | 10.79 |
| agent on the opponent side | 71.4% (370 / 518) | 74.6% (468 / 627) |
| mean depth, opponent side | 11.43 | 11.43 |

## Prep coverage (repertoire side to move, prep tree has a repertoire move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| positions | 36.8% (3,761 / 10,223) | 37.4% (4,606 / 12,303) |

## Move quality: agreement with the master move (every held-out position)

| | Ruy Lopez, all | Ruy Lopez, strict | Sicilian Taimanov, all | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 20,427 | 7,114 | 24,631 | 9,431 |
| Top-1 agreement | 59.2% | 30.8% | 59.7% | 30.0% |
| Top-3 agreement | 80.5% | 54.0% | 77.7% | 52.7% |

Sanity checks over 45,058 positions: 0 policies containing an illegal move, 0 policies that are empty or do not sum to 1, 0 illegal master moves.

## Move quality: Stockfish depth 14 (seeded sample of distinct positions, seed 0)

| | Ruy Lopez, seen | Ruy Lopez, strict | Sicilian Taimanov, seen | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 500 | 500 | 500 | 500 |
| out of book (fallback) | 0.0% | 0.0% | 0.0% | 0.0% |
| mean CPL, top move | 12.8 | 85.9 | 19.9 | 103.9 |
| median CPL, top move | 0.0 | 19.0 | 0.0 | 33.5 |
| blunder rate (CPL >= 100), top move | 2.4% | 21.8% | 4.6% | 26.2% |
| mean WDL expectation loss, top move | 0.021 | 0.133 | 0.038 | 0.149 |
| mean CPL, sampled | 34.3 | 128.8 | 40.8 | 152.1 |
| median CPL, sampled | 15.6 | 94.5 | 19.0 | 120.8 |
| blunder rate (CPL >= 100), sampled | 7.8% | 33.2% | 10.1% | 39.6% |
| mean WDL expectation loss, sampled | 0.051 | 0.185 | 0.070 | 0.208 |
| mean CPL, master | 4.2 | 9.8 | 7.9 | 13.8 |
| median CPL, master | 0.0 | 0.0 | 0.0 | 0.0 |
| blunder rate (CPL >= 100), master | 0.0% | 0.4% | 0.2% | 1.8% |
| mean WDL expectation loss, master | 0.008 | 0.019 | 0.020 | 0.032 |

`top move`: the agent's most probable move (a uniform fallback is one seeded random legal move). `sampled`: expectation over the agent's whole policy. `master`: the move actually played, for reference. For `sampled`, the median is over per-position expected CPL.
