"""Repertoire tree from the prep PGNs: every move the prep files give at each position."""

from collections.abc import Iterable
from pathlib import Path

import chess.pgn

from chess_trainer.data.ingest import read_games
from chess_trainer.data.openings import classify, fen_norm

# Chessable exports escape parentheses inside comments.
_ESCAPES = {"@@StartBracket@@": "(", "@@EndBracket@@": ")"}


def _clean(comment: str) -> str:
    for token, char in _ESCAPES.items():
        comment = comment.replace(token, char)
    return " ".join(comment.split())


def _walk(game: chess.pgn.Game, opening: str, chapter: str, tree: dict) -> None:
    stack: list[chess.pgn.GameNode] = [game]
    while stack:
        parent = stack.pop()
        board = parent.board()
        moves = tree.setdefault(fen_norm(board), {}) if parent.variations else {}
        for i, child in enumerate(parent.variations):
            entry = moves.setdefault(
                child.move.uci(),
                {
                    "san": board.san(child.move),
                    "main": False,
                    "opening": opening,
                    "chapters": [],
                    "comment": "",
                },
            )
            entry["main"] = entry["main"] or i == 0
            if chapter not in entry["chapters"]:
                entry["chapters"].append(chapter)
            if not entry["comment"]:
                entry["comment"] = _clean(child.comment)
            stack.append(child)


def build_tree(paths: Iterable[Path]) -> tuple[dict, dict]:
    """Returns ({fen_norm: {move_uci: info}}, stats). `main` marks a chapter's first
    choice at that position; the other moves are side variations the prep file discusses."""
    tree: dict = {}
    stats: dict = {"chapters": 0, "kept": {}, "skipped_fen": 0, "skipped_off_root": 0}
    for path in paths:
        for game in read_games(Path(path)):
            stats["chapters"] += 1
            if "FEN" in game.headers:
                stats["skipped_fen"] += 1
                continue
            opening = classify(game.mainline_moves())
            if opening is None:
                stats["skipped_off_root"] += 1
                continue
            stats["kept"][opening] = stats["kept"].get(opening, 0) + 1
            _walk(game, opening, game.headers.get("White", "?"), tree)
    return tree, stats
