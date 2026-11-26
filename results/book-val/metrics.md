# Opening coverage: book agent, val split

Git `3626f62`, seed 0. Definitions are in `chess_trainer/eval/coverage.py`.

## Position coverage (non-fallback move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| all positions | 65.0% (13,226 / 20,331) | 62.6% (15,440 / 24,676) |
| strict positions (never seen in train) | 0.0% (0 / 7,105) | 0.0% (0 / 9,236) |

## Line coverage (at least 10 consecutive known own moves)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| agent on the repertoire side | 81.3% (421 / 518) | 75.4% (471 / 625) |
| mean depth, repertoire side | 12.82 | 11.84 |
| agent on the opponent side | 74.5% (386 / 518) | 80.9% (507 / 627) |
| mean depth, opponent side | 12.29 | 12.37 |

## Prep coverage (repertoire side to move, prep tree has a repertoire move)

| | Ruy Lopez | Sicilian Taimanov |
|---|---:|---:|
| positions | 35.9% (3,651 / 10,177) | 37.4% (4,610 / 12,325) |

## Move quality: agreement with the master move (every held-out position)

| | Ruy Lopez, all | Ruy Lopez, strict | Sicilian Taimanov, all | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 20,331 | 7,105 | 24,676 | 9,236 |
| Top-1 agreement | 51.8% | 3.3% | 52.0% | 3.2% |
| Top-3 agreement | 64.8% | 9.5% | 61.6% | 9.2% |

Sanity checks over 45,007 positions: 0 policies containing an illegal move, 0 policies that are empty or do not sum to 1, 0 illegal master moves.

## Move quality: Stockfish depth 14 (seeded sample of distinct positions, seed 0)

| | Ruy Lopez, seen | Ruy Lopez, strict | Sicilian Taimanov, seen | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| positions | 0 | 0 | 0 | 0 |
| out of book (fallback) | n/a | n/a | n/a | n/a |
| mean CPL, top move | n/a | n/a | n/a | n/a |
| median CPL, top move | n/a | n/a | n/a | n/a |
| blunder rate (CPL >= 100), top move | n/a | n/a | n/a | n/a |
| mean WDL expectation loss, top move | n/a | n/a | n/a | n/a |
| mean CPL, sampled | n/a | n/a | n/a | n/a |
| median CPL, sampled | n/a | n/a | n/a | n/a |
| blunder rate (CPL >= 100), sampled | n/a | n/a | n/a | n/a |
| mean WDL expectation loss, sampled | n/a | n/a | n/a | n/a |
| mean CPL, master | n/a | n/a | n/a | n/a |
| median CPL, master | n/a | n/a | n/a | n/a |
| blunder rate (CPL >= 100), master | n/a | n/a | n/a | n/a |
| mean WDL expectation loss, master | n/a | n/a | n/a | n/a |

`top move`: the agent's most probable move (a uniform fallback is one seeded random legal move). `sampled`: expectation over the agent's whole policy. `master`: the move actually played, for reference. For `sampled`, the median is over per-position expected CPL.
