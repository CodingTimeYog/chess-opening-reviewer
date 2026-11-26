"""Build the retrieval indexes: uv run python -m chess_trainer.rag --checkpoint resnet_main

Writes the policy-network embedding of every distinct indexed position to
data/processed/retrieval_train.npz (train games, used by the eval) and
data/processed/retrieval_all.npz (every game, used by the app).
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from chess_trainer import config
from chess_trainer.agents.policy import PolicyAgent
from chess_trainer.rag import Retriever, checkpoint_signature, index_path, load_or_build_index


def resolve_checkpoint(name: str) -> Path:
    path = Path(name)
    return path if path.exists() else config.CHECKPOINT_DIR / f"{name}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the embedding retrieval indexes.")
    parser.add_argument("--checkpoint", default="resnet_main")
    parser.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    agent = PolicyAgent.from_checkpoint(checkpoint)
    signature = checkpoint_signature(checkpoint)
    positions = pd.read_parquet(args.data / "positions.parquet")
    for splits in (("train",), None):
        tick = time.perf_counter()
        retriever = Retriever(positions, splits=splits)
        path = index_path(splits)
        how = load_or_build_index(retriever, agent, path, signature)
        games = retriever.rows["game_id"].nunique()
        print(
            f"{path.name}: {how}, {len(retriever.indexed_positions()):,} positions from "
            f"{games:,} games, {path.stat().st_size / 1e6:.1f} MB, "
            f"{time.perf_counter() - tick:.0f}s"
        )


if __name__ == "__main__":
    main()
