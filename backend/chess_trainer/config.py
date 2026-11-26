"""Paths and constants shared across the package."""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CHESS_TRAINER_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = REPO_ROOT / "results"

STOCKFISH_PATH = Path(
    os.environ.get(
        "STOCKFISH_PATH",
        "C:/Users/aswin/Desktop/Chess_Engines/stockfish/stockfish-windows-x86-64-avx2.exe",
    )
)
ENGINE_DEPTH = 14
ENGINE_WORKERS = max(1, (os.cpu_count() or 4) - 4)  # one single-threaded engine each

MAX_PLY = 40  # opening phase ends at move 20
ROOT_WINDOW = 12  # a game must reach its opening's root position within this many plies
SEED = 0


def git_sha() -> str:
    """Short sha of HEAD, recorded with every data build and eval run."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip()
