"""Feedback correctness on a labelled set of (position, player move).

uv run python -m chess_trainer.eval.feedback

Cases: for every position in the seeded held-out sample, up to three player moves: the
master's move, the engine's best move, and a seeded random legal move (duplicates dropped).
Ground truth comes from the offline cache, where Stockfish scored every legal move:
the move's centipawn loss, its class (good / inaccuracy / mistake / blunder) and the set of
best moves.

The system under test is the live feedback path: two short searches per move, the facts
built from them, and the template text. It never reads the offline scores. Checked:
- classification accuracy: the feedback's class equals the ground-truth class;
- best move named correctly: the named best move is one of the ground-truth best moves
  (also reported: within 10 cp of the best);
- text consistency: the text names the same best move and class as the facts;
- hallucinated citations: a cited game that does not exist or never reached the position.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import pandas as pd

from chess_trainer import config
from chess_trainer.agents.book import BookAgent
from chess_trainer.data.openings import OPENINGS, board_from
from chess_trainer.data.repertoire import load_tree
from chess_trainer.engine import EvalCache, LiveAnalyser, clipped_cp, evaluate_positions
from chess_trainer.eval.quality import SAMPLE_PER_STRATUM, sample_positions
from chess_trainer.feedback import FeedbackBuilder, classify_loss
from chess_trainer.rag import Retriever

KINDS = ("master", "best", "random")
CLASSES = ("good", "inaccuracy", "mistake", "blunder")
NEAR_CP = 10


def build_cases(sample: pd.DataFrame, evals: dict, seed: int = config.SEED) -> pd.DataFrame:
    cases = []
    for row in sample.itertuples():
        scores = {move: clipped_cp(*score) for move, score in evals[row.fen_norm].items()}
        best_cp = max(scores.values())
        best_moves = sorted(move for move, cp in scores.items() if cp == best_cp)
        rng = random.Random(f"{seed}:{row.fen_norm}")
        candidates = {
            "master": row.move_uci,
            "best": best_moves[0],
            "random": rng.choice(sorted(scores)),
        }
        seen = set()
        for kind in KINDS:
            move = candidates[kind]
            if move in seen:
                continue
            seen.add(move)
            loss = best_cp - scores[move]
            cases.append(
                {
                    "opening": row.opening,
                    "stratum": row.stratum,
                    "fen_norm": row.fen_norm,
                    "ply": row.ply,
                    "move_uci": move,
                    "kind": kind,
                    "truth_cp_loss": loss,
                    "truth_class": classify_loss(loss),
                    "truth_best": best_moves,
                }
            )
    return pd.DataFrame(cases)


def _text_consistent(text: str, facts: dict) -> bool:
    label = facts["classification"]
    names_class = (
        ("first choice" in text or "is fine" in text) if label == "good" else label in text
    )
    return facts["best"]["san"] in text and names_class


def evaluate_feedback(
    cases: pd.DataFrame, builder: FeedbackBuilder, positions: pd.DataFrame, evals: dict
) -> dict:
    reached = set(zip(positions["game_id"], positions["fen_norm"], strict=True))
    records = []
    for i, case in enumerate(cases.itertuples(), start=1):
        board = board_from(case.fen_norm, case.ply)
        feedback = builder.move_feedback(board, chess.Move.from_uci(case.move_uci))
        facts = feedback["facts"]
        scores = {move: clipped_cp(*score) for move, score in evals[case.fen_norm].items()}
        named = facts["best"]["uci"]
        cited = feedback["citations"]
        records.append(
            {
                "opening": case.opening,
                "stratum": case.stratum,
                "kind": case.kind,
                "truth_class": case.truth_class,
                "predicted_class": facts["classification"],
                "class_ok": facts["classification"] == case.truth_class,
                "best_ok": named in case.truth_best,
                "best_near": max(scores.values()) - scores.get(named, -10_000) <= NEAR_CP,
                "text_ok": _text_consistent(feedback["text"], facts),
                "cited": len(cited),
                "hallucinated": sum((c["game_id"], case.fen_norm) not in reached for c in cited),
            }
        )
        if i % 500 == 0:
            print(f"feedback: {i:,} / {len(cases):,} cases", file=sys.stderr, flush=True)
    frame = pd.DataFrame(records)

    def block(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n": 0}
        return {
            "n": len(sub),
            "classification_accuracy": round(100 * sub["class_ok"].mean(), 2),
            "best_move_correct": round(100 * sub["best_ok"].mean(), 2),
            "best_move_within_10cp": round(100 * sub["best_near"].mean(), 2),
            "text_consistent": round(100 * sub["text_ok"].mean(), 2),
            "citations": int(sub["cited"].sum()),
            "hallucinated_citations": int(sub["hallucinated"].sum()),
        }

    confusion = (
        frame.groupby(["truth_class", "predicted_class"]).size().unstack(fill_value=0)
        if len(frame)
        else pd.DataFrame()
    )
    return {
        "overall": block(frame),
        "by_stratum": {
            oid: {
                s: block(frame[(frame["opening"] == oid) & (frame["stratum"] == s)])
                for s in ("seen", "strict")
            }
            for oid in OPENINGS
        },
        "by_kind": {kind: block(frame[frame["kind"] == kind]) for kind in KINDS},
        "by_truth_class": {c: block(frame[frame["truth_class"] == c]) for c in CLASSES},
        "confusion": {
            truth: {pred: int(confusion.loc[truth, pred]) for pred in confusion.columns}
            for truth in confusion.index
        },
    }


def feedback_markdown(metrics: dict, meta: dict) -> str:
    keys = [
        ("cases", "n", ","),
        ("classification accuracy", "classification_accuracy", ".1f"),
        ("best move named correctly", "best_move_correct", ".1f"),
        ("best move within 10 cp of best", "best_move_within_10cp", ".1f"),
        ("text consistent with facts", "text_consistent", ".1f"),
        ("citations checked", "citations", ","),
        ("hallucinated citations", "hallucinated_citations", ","),
    ]

    def table(columns: list[tuple[str, dict]]) -> str:
        head = "| | " + " | ".join(name for name, _ in columns) + " |\n"
        head += "|---|" + "---:|" * len(columns) + "\n"
        body = ""
        for label, key, spec in keys:
            cells = []
            for _, block in columns:
                value = block.get(key)
                percent = "%" if spec == ".1f" else ""
                cells.append("n/a" if value is None else f"{value:{spec}}{percent}")
            body += f"| {label} | " + " | ".join(cells) + " |\n"
        return head + body

    strata = [
        (f"{OPENINGS[oid].name}, {s}", metrics["by_stratum"][oid][s])
        for oid in OPENINGS
        for s in ("seen", "strict")
    ]
    classes = [c for c in CLASSES if metrics["by_truth_class"][c]["n"]]
    confusion = metrics["confusion"]
    predicted = sorted({p for row in confusion.values() for p in row}, key=CLASSES.index)
    matrix = "| truth \\ feedback | " + " | ".join(predicted) + " |\n"
    matrix += "|---|" + "---:|" * len(predicted) + "\n"
    for truth in classes:
        cells = " | ".join(f"{confusion.get(truth, {}).get(p, 0):,}" for p in predicted)
        matrix += f"| {truth} | {cells} |\n"
    return "".join(
        [
            f"# Feedback correctness: {meta['layer']} layer, {meta['split']} split\n\n",
            f"Git `{meta['git_sha']}`, seed {meta['seed']}, Stockfish depth {meta['depth']}. "
            "Definitions are in `chess_trainer/eval/feedback.py`.\n\n",
            "## Overall and by kind of player move\n\n",
            table([("all", metrics["overall"])] + [(k, metrics["by_kind"][k]) for k in KINDS]),
            "\n## By opening and by seen / strict position\n\n",
            table(strata),
            "\n## By ground-truth class\n\n",
            table([(c, metrics["by_truth_class"][c]) for c in classes]),
            "\n## Confusion matrix (rows: ground truth, columns: class given in the feedback)\n\n",
            matrix,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate feedback correctness.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--sample", type=int, default=SAMPLE_PER_STRATUM)
    parser.add_argument("--depth", type=int, default=config.ENGINE_DEPTH)
    parser.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    positions = pd.read_parquet(args.data / "positions.parquet")
    tree = load_tree(args.data / "repertoire.json")
    sample = sample_positions(
        positions, split=args.split, n_per_stratum=args.sample, seed=args.seed
    )
    cache = EvalCache()
    evals = evaluate_positions(sample["fen_norm"], depth=args.depth, cache=cache)
    cases = build_cases(sample, evals, seed=args.seed)

    analyser = LiveAnalyser(cache, depth=args.depth, use_full_cache=False)
    builder = FeedbackBuilder(
        BookAgent.from_positions(positions), tree, Retriever(positions, splits=None), analyser
    )
    try:
        metrics = evaluate_feedback(cases, builder, positions, evals)
    finally:
        analyser.close()

    meta = {
        "layer": "template",
        "split": args.split,
        "seed": args.seed,
        "depth": args.depth,
        "git_sha": config.git_sha(),
    }
    run_dir = args.results / f"feedback-{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({**meta, **metrics}, indent=2), encoding="utf-8"
    )
    report = feedback_markdown(metrics, meta)
    (run_dir / "metrics.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
