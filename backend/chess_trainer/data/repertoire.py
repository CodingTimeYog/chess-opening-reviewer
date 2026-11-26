"""Repertoire tree from the prep PGNs.

Every move the prep files discuss is stored, with `in_repertoire` saying whether it is
part of the repertoire:

- Repertoire side (White in the Ruy Lopez, Black in the Taimanov): only a chapter's first
  choice at a position counts, and never a move annotated ?, ?? or ?!.
- Opponent side: every branch counts, so the trainer covers all opponent replies.

A move only counts when the position it is played from is itself on the repertoire path;
lines below a rejected repertoire-side move are kept for reference but are off the path.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import chess
import chess.pgn

from chess_trainer.data.ingest import read_games
from chess_trainer.data.openings import OPENINGS, classify, fen_norm

# Chessable exports escape parentheses inside comments.
_ESCAPES = {"@@StartBracket@@": "(", "@@EndBracket@@": ")"}
DUBIOUS_NAGS = {chess.pgn.NAG_MISTAKE, chess.pgn.NAG_BLUNDER, chess.pgn.NAG_DUBIOUS_MOVE}


def _clean(comment: str) -> str:
    for token, char in _ESCAPES.items():
        comment = comment.replace(token, char)
    return " ".join(comment.split())


def _walk(game: chess.pgn.Game, opening: str, chapter: str, tree: dict) -> None:
    side = OPENINGS[opening].side
    stack: list[tuple[chess.pgn.GameNode, bool]] = [(game, True)]
    while stack:
        parent, on_path = stack.pop()
        if not parent.variations:
            continue
        board = parent.board()
        own = board.turn == side
        moves = tree.setdefault(fen_norm(board), {})
        for i, child in enumerate(parent.variations):
            entry = moves.setdefault(
                child.move.uci(),
                {
                    "san": board.san(child.move),
                    "side": "own" if own else "opp",
                    "main": False,
                    "in_repertoire": False,
                    "nags": [],
                    "opening": opening,
                    "chapters": [],
                    "comment": "",
                },
            )
            is_main = i == 0
            if own:
                in_repertoire = on_path and is_main and not (child.nags & DUBIOUS_NAGS)
            else:
                in_repertoire = on_path
            entry["main"] = entry["main"] or is_main
            entry["in_repertoire"] = entry["in_repertoire"] or in_repertoire
            entry["nags"] = sorted(set(entry["nags"]) | child.nags)
            if chapter not in entry["chapters"]:
                entry["chapters"].append(chapter)
            if not entry["comment"]:
                entry["comment"] = _clean(child.comment)
            stack.append((child, in_repertoire))


def build_tree(paths: Iterable[Path]) -> tuple[dict, dict]:
    """Returns ({fen_norm: {move_uci: info}}, stats)."""
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


def repertoire_moves(tree: dict, fen: str) -> set[str]:
    """Moves that are in repertoire from this position (empty when the tree is silent)."""
    return {uci for uci, info in tree.get(fen, {}).items() if info["in_repertoire"]}


def load_tree(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
