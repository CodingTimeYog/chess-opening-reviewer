"""Run the evaluation harness: uv run python -m chess_trainer.eval --agent book --split test"""

import argparse
import json
from pathlib import Path

import pandas as pd

from chess_trainer import config
from chess_trainer.agents.book import BookAgent
from chess_trainer.data.repertoire import load_tree
from chess_trainer.engine import EvalCache, evaluate_positions
from chess_trainer.eval.coverage import evaluate_coverage, metrics_markdown
from chess_trainer.eval.quality import (
    SAMPLE_PER_STRATUM,
    agreement,
    engine_quality,
    quality_markdown,
    sample_positions,
)

AGENTS = {
    "book": lambda positions, seed: BookAgent.from_positions(positions, seed=seed),
    "book-weighted": lambda positions, seed: BookAgent.from_positions(
        positions, result_weighted=True, seed=seed
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent on held-out positions.")
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--depth", type=int, default=config.ENGINE_DEPTH)
    parser.add_argument(
        "--sample", type=int, default=SAMPLE_PER_STRATUM, help="positions per opening and stratum"
    )
    parser.add_argument("--workers", type=int, default=config.ENGINE_WORKERS)
    args = parser.parse_args()

    positions = pd.read_parquet(args.data / "positions.parquet")
    tree = load_tree(args.data / "repertoire.json")
    agent = AGENTS[args.agent](positions, args.seed)
    sha = config.git_sha()

    coverage = evaluate_coverage(
        agent, positions, tree, split=args.split, seed=args.seed, git_sha=sha
    )
    sample = sample_positions(
        positions, split=args.split, n_per_stratum=args.sample, seed=args.seed
    )
    evals = evaluate_positions(
        sample["fen_norm"], depth=args.depth, cache=EvalCache(), workers=args.workers
    )
    quality = {
        "depth": args.depth,
        "sample_seed": args.seed,
        "sample_per_stratum": args.sample,
        "agreement": agreement(agent, positions, split=args.split),
        "engine": engine_quality(agent, sample, evals, seed=args.seed),
    }
    metrics = {
        "agent": args.agent,
        "split": args.split,
        "seed": args.seed,
        "git_sha": sha,
        "coverage": coverage,
        "quality": quality,
    }

    run_dir = args.results / f"{args.agent}-{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = metrics_markdown(coverage) + "\n" + quality_markdown(quality)
    (run_dir / "metrics.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
