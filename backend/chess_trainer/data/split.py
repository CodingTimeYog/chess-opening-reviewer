"""Game-level train/val/test split and the strict held-out subset."""

import random

import pandas as pd

TRAIN_FRACTION, VAL_FRACTION = 0.8, 0.1  # remainder is test


def assign_splits(games: pd.DataFrame, seed: int) -> pd.Series:
    """Split label per game. Every position inherits its game's label, so no game can
    leak across splits. Shuffled within each opening so each one gets 80/10/10."""
    labels: dict[str, str] = {}
    for opening, group in games.groupby("opening"):
        ids = sorted(group["game_id"])
        random.Random(f"{seed}:{opening}").shuffle(ids)
        n_train = round(len(ids) * TRAIN_FRACTION)
        n_val = round(len(ids) * VAL_FRACTION)
        for i, game_id in enumerate(ids):
            labels[game_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return games["game_id"].map(labels).rename("split")


def mark_strict(positions: pd.DataFrame) -> pd.Series:
    """True for val/test rows whose position never occurs in train."""
    train_fens = set(positions.loc[positions["split"] == "train", "fen_norm"])
    held_out = positions["split"] != "train"
    return (held_out & ~positions["fen_norm"].isin(train_fens)).rename("strict")
