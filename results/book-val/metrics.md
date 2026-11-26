# Opening coverage: book agent, val split

Git `c14b485`, seed 0. Definitions are in `chess_trainer/eval/coverage.py`.

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
