# Retrieval relevance, test split

Git `8c86664`, seed 0. Index: train games only. Definitions are in `chess_trainer/eval/retrieval.py`.

| | Ruy Lopez, seen | Ruy Lopez, strict | Sicilian Taimanov, seen | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| query positions | 500 | 500 | 500 | 500 |
| precision@5, random | 4.3% | 0.0% | 4.4% | 0.3% |
| MRR, random | 0.067 | 0.002 | 0.066 | 0.008 |
| mean games returned, random | 5.00 | 5.00 | 5.00 | 5.00 |
| precision@5, exact | 79.8% | 0.0% | 78.4% | 0.0% |
| MRR, exact | 1.000 | 0.000 | 1.000 | 0.000 |
| mean games returned, exact | 3.99 | 0.00 | 3.92 | 0.00 |
