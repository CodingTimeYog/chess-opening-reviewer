"""Retrieval relevance on held-out positions.

uv run python -m chess_trainer.eval.retrieval [--checkpoint resnet_main]

Queries are the seeded sample of distinct held-out positions (seen and strict, per opening)
used for move quality. The index holds train games only. A retrieved game is relevant if it
reached the exact query position, or if it shares the query game's ECO code and, from the
position that matched, the master played the same next move as in the query game.

Reported per method: precision@5 (relevant hits / 5, so returning fewer than five costs
precision) and MRR (1 / rank of the first relevant hit, 0 if none).
Methods: `random` (same opening and ply, the control), `exact` (position lookup only) and,
with a checkpoint, `hybrid` (exact, then nearest neighbours by policy embedding).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from chess_trainer import config
from chess_trainer.data.openings import OPENINGS
from chess_trainer.eval.quality import SAMPLE_PER_STRATUM, sample_positions
from chess_trainer.rag import Hit, RandomRetriever, Retriever, index_embeddings

K = 5


def is_relevant(hit: Hit, query_eco: str, master_move: str) -> bool:
    return hit.exact or (hit.eco == query_eco and hit.move_uci == master_move)


def evaluate_retrieval(retriever, sample: pd.DataFrame, k: int = K) -> dict:
    records = []
    for row in sample.itertuples():
        hits = retriever.retrieve(row.fen_norm, k=k, ply=row.ply, opening=row.opening)
        relevant = [is_relevant(hit, row.eco, row.move_uci) for hit in hits[:k]]
        first = next((i for i, ok in enumerate(relevant) if ok), None)
        records.append(
            {
                "opening": row.opening,
                "stratum": row.stratum,
                "precision": sum(relevant) / k,
                "rr": 0.0 if first is None else 1 / (first + 1),
                "returned": len(hits),
            }
        )
    frame = pd.DataFrame(records, columns=["opening", "stratum", "precision", "rr", "returned"])

    def block(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n": 0, "precision": None, "mrr": None, "mean_returned": None}
        return {
            "n": len(sub),
            "precision": round(100 * sub["precision"].mean(), 2),
            "mrr": round(float(sub["rr"].mean()), 4),
            "mean_returned": round(float(sub["returned"].mean()), 2),
        }

    openings = {}
    for oid in OPENINGS:
        sub = frame[frame["opening"] == oid]
        openings[oid] = {s: block(sub[sub["stratum"] == s]) for s in ("seen", "strict")}
    return {"k": k, "openings": openings}


def retrieval_markdown(results: dict[str, dict], meta: dict) -> str:
    k = next(iter(results.values()))["k"]
    columns = [(oid, s) for oid in OPENINGS for s in ("seen", "strict")]
    head = "| | " + " | ".join(f"{OPENINGS[o].name}, {s}" for o, s in columns) + " |\n"
    head += "|---|" + "---:|" * len(columns) + "\n"

    def row(label: str, metrics: dict, key: str, spec: str, suffix: str = "") -> str:
        cells = []
        for oid, stratum in columns:
            value = metrics["openings"][oid][stratum][key]
            cells.append("n/a" if value is None else f"{value:{spec}}{suffix}")
        return f"| {label} | " + " | ".join(cells) + " |\n"

    first = next(iter(results.values()))
    out = [
        f"# Retrieval relevance, {meta['split']} split\n\n",
        f"Git `{meta['git_sha']}`, seed {meta['seed']}. Index: train games only. "
        "Definitions are in `chess_trainer/eval/retrieval.py`.\n\n",
        head,
        row("query positions", first, "n", ","),
    ]
    for name, metrics in results.items():
        out.append(row(f"precision@{k}, {name}", metrics, "precision", ".1f", "%"))
        out.append(row(f"MRR, {name}", metrics, "mrr", ".3f"))
        out.append(row(f"mean games returned, {name}", metrics, "mean_returned", ".2f"))
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate master-game retrieval.")
    parser.add_argument("--checkpoint", help="policy checkpoint for the embedding fallback")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--sample", type=int, default=SAMPLE_PER_STRATUM)
    parser.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    positions = pd.read_parquet(args.data / "positions.parquet")
    sample = sample_positions(
        positions, split=args.split, n_per_stratum=args.sample, seed=args.seed
    )
    results = {
        "random": evaluate_retrieval(RandomRetriever(positions, seed=args.seed), sample),
        "exact": evaluate_retrieval(Retriever(positions), sample),
    }
    if args.checkpoint:
        from chess_trainer.agents.policy import PolicyAgent  # imports torch

        path = Path(args.checkpoint)
        if not path.exists():
            path = config.CHECKPOINT_DIR / f"{args.checkpoint}.pt"
        retriever = Retriever(positions)
        index_embeddings(retriever, PolicyAgent.from_checkpoint(path))
        results["hybrid"] = evaluate_retrieval(retriever, sample)

    meta = {"split": args.split, "seed": args.seed, "git_sha": config.git_sha()}
    run_dir = args.results / f"retrieval-{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({**meta, "methods": results}, indent=2), encoding="utf-8"
    )
    report = retrieval_markdown(results, meta)
    (run_dir / "metrics.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
