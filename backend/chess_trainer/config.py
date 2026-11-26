"""Paths and constants shared across the package."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CHESS_TRAINER_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

MAX_PLY = 40  # opening phase ends at move 20
ROOT_WINDOW = 12  # a game must reach its opening's root position within this many plies
SEED = 0
