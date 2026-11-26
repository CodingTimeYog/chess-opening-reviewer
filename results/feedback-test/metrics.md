# Feedback correctness: template layer, test split

Git `aea6697`, seed 0, Stockfish depth 14. Definitions are in `chess_trainer/eval/feedback.py`.

## Overall and by kind of player move

| | all | master | best | random |
|---|---:|---:|---:|---:|
| cases | 4,829 | 2,000 | 917 | 1,912 |
| classification accuracy | 94.8% | 97.0% | 99.5% | 90.3% |
| best move named correctly | 69.6% | 72.8% | 55.7% | 72.9% |
| best move within 10 cp of best | 91.0% | 91.8% | 87.9% | 91.7% |
| text consistent with facts | 100.0% | 100.0% | 100.0% | 100.0% |
| citations checked | 9,393 | 3,906 | 1,749 | 3,738 |
| hallucinated citations | 0 | 0 | 0 | 0 |

## By opening and by seen / strict position

| | Ruy Lopez, seen | Ruy Lopez, strict | Sicilian Taimanov, seen | Sicilian Taimanov, strict |
|---|---:|---:|---:|---:|
| cases | 1,175 | 1,210 | 1,219 | 1,225 |
| classification accuracy | 95.7% | 94.7% | 95.5% | 93.2% |
| best move named correctly | 69.1% | 69.1% | 69.6% | 70.6% |
| best move within 10 cp of best | 92.9% | 90.8% | 92.0% | 88.2% |
| text consistent with facts | 100.0% | 100.0% | 100.0% | 100.0% |
| citations checked | 3,385 | 1,233 | 3,502 | 1,273 |
| hallucinated citations | 0 | 0 | 0 | 0 |

## By ground-truth class

| | good | inaccuracy | mistake | blunder |
|---|---:|---:|---:|---:|
| cases | 3,169 | 385 | 350 | 925 |
| classification accuracy | 97.3% | 73.2% | 86.6% | 98.0% |
| best move named correctly | 66.2% | 72.5% | 71.4% | 79.3% |
| best move within 10 cp of best | 90.4% | 90.1% | 90.6% | 93.5% |
| text consistent with facts | 100.0% | 100.0% | 100.0% | 100.0% |
| citations checked | 6,256 | 745 | 693 | 1,699 |
| hallucinated citations | 0 | 0 | 0 | 0 |

## Confusion matrix (rows: ground truth, columns: class given in the feedback)

| truth \ feedback | good | inaccuracy | mistake | blunder |
|---|---:|---:|---:|---:|
| good | 3,085 | 82 | 2 | 0 |
| inaccuracy | 62 | 282 | 40 | 1 |
| mistake | 2 | 27 | 303 | 18 |
| blunder | 0 | 0 | 18 | 907 |
