"""Master-game PGNs -> one row per game and one row per position."""

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

from chess_trainer.config import MAX_PLY, SEED
from chess_trainer.data.openings import OPENINGS, classify, fen_norm
from chess_trainer.data.split import assign_splits, mark_strict

__all__ = ["STAGES", "build_corpus", "fen_norm", "read_games"]

STAGES = ("reached_root", "both_rated", "rating_ok", "has_result", "unique")
RESULTS = {"1-0", "0-1", "1/2-1/2"}
GAME_COLUMNS = [
    "game_id",
    "opening",
    "result",
    "white_elo",
    "black_elo",
    "eco",
    "source",
    "event",
    "year",
    "white",
    "black",
    "n_plies",
]


def read_games(path: Path) -> Iterator[chess.pgn.Game]:
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            yield game


def _elo(value: str | None) -> int | None:
    return int(value) if value and value.isdigit() and int(value) > 0 else None


def _year(date: str) -> int | None:
    return int(date[:4]) if date[:4].isdigit() else None


def _game_id(headers: chess.pgn.Headers, ucis: list[str]) -> str:
    """Stable id from players, date and moves; identical games collide, which dedupes them."""
    parts = [headers.get("White", ""), headers.get("Black", ""), headers.get("Date", "")]
    key = "|".join([*parts, " ".join(ucis)])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def build_corpus(
    paths: Iterable[Path], max_ply: int = MAX_PLY, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse, classify, filter and split. Returns (games, positions, stats)."""
    stats: dict = {
        "parsed": 0,
        "parse_errors": 0,
        "non_standard_start": 0,
        "openings": {oid: {"stages": dict.fromkeys(STAGES, 0)} for oid in OPENINGS},
    }
    games: list[dict] = []
    moves_by_game: dict[str, list[tuple[str, str]]] = {}

    for path in paths:
        for game in read_games(Path(path)):
            stats["parsed"] += 1
            h = game.headers
            if game.errors:
                stats["parse_errors"] += 1
                continue
            if "FEN" in h or h.get("Variant", "Standard").lower() not in ("standard", "chess"):
                stats["non_standard_start"] += 1
                continue
            moves = list(game.mainline_moves())
            opening = classify(moves)
            if opening is None:
                continue
            stage = stats["openings"][opening]["stages"]
            stage["reached_root"] += 1
            white_elo, black_elo = _elo(h.get("WhiteElo")), _elo(h.get("BlackElo"))
            if white_elo is None or black_elo is None:
                continue
            stage["both_rated"] += 1
            if min(white_elo, black_elo) < OPENINGS[opening].min_elo:
                continue
            stage["rating_ok"] += 1
            if h.get("Result") not in RESULTS:
                continue
            stage["has_result"] += 1
            game_id = _game_id(h, [m.uci() for m in moves])
            if game_id in moves_by_game:
                continue
            stage["unique"] += 1

            board, rows = game.board(), []
            for move in moves[:max_ply]:
                rows.append((fen_norm(board), move.uci()))
                board.push(move)
            moves_by_game[game_id] = rows
            games.append(
                {
                    "game_id": game_id,
                    "opening": opening,
                    "result": h["Result"],
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "eco": h.get("ECO", ""),
                    "source": Path(path).stem,
                    "event": h.get("Event", ""),
                    "year": _year(h.get("Date", "")),
                    "white": h.get("White", ""),
                    "black": h.get("Black", ""),
                    "n_plies": len(moves),
                }
            )

    games_df = pd.DataFrame(games, columns=GAME_COLUMNS).astype({"year": "Int64"})
    games_df["split"] = assign_splits(games_df, seed=seed)

    flat = [
        (game_id, ply, fen, uci)
        for game_id, rows in moves_by_game.items()
        for ply, (fen, uci) in enumerate(rows)
    ]
    pos = pd.DataFrame(flat, columns=["game_id", "ply", "fen_norm", "move_uci"])
    positions_df = pos.merge(games_df.drop(columns="n_plies"), on="game_id", validate="m:1")
    positions_df["strict"] = mark_strict(positions_df)

    stats["seed"] = seed
    stats["max_ply"] = max_ply
    for oid, entry in stats["openings"].items():
        entry["min_elo"] = OPENINGS[oid].min_elo
        g = games_df[games_df["opening"] == oid]
        p = positions_df[positions_df["opening"] == oid]
        entry["unique_positions"] = int(p["fen_norm"].nunique())
        entry["splits"] = {
            s: {
                "games": int((g["split"] == s).sum()),
                "positions": int((p["split"] == s).sum()),
                "strict_positions": int(((p["split"] == s) & p["strict"]).sum()),
            }
            for s in ("train", "val", "test")
        }
    return games_df, positions_df, stats
