# Opening coverage: book agent, test split

Git `c14b485`, seed 0. Definitions are in `chess_trainer/eval/coverage.py`.

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
