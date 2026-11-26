"""Repertoire roots. A game belongs to an opening when it reaches the root position."""

from collections.abc import Iterable
from dataclasses import dataclass

import chess

from chess_trainer.config import ROOT_WINDOW


def fen_norm(board: chess.Board) -> str:
    """FEN without move counters; en-passant square only when a capture is legal."""
    return board.epd()


@dataclass(frozen=True)
class Opening:
    id: str
    name: str
    side: chess.Color  # the side whose repertoire this is
    root_san: tuple[str, ...]
    min_elo: int  # both players must be rated at least this


OPENINGS: dict[str, Opening] = {
    o.id: o
    for o in (
        Opening("ruy_lopez", "Ruy Lopez", chess.WHITE, ("e4", "e5", "Nf3", "Nc6", "Bb5"), 2500),
        Opening(
            "taimanov",
            "Sicilian Taimanov",
            chess.BLACK,
            ("e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "Nc6"),
            2400,
        ),
    )
}


def _root_fen(opening: Opening) -> str:
    board = chess.Board()
    for san in opening.root_san:
        board.push_san(san)
    return fen_norm(board)


ROOT_FENS: dict[str, str] = {o.id: _root_fen(o) for o in OPENINGS.values()}
_FEN_TO_ID = {fen: oid for oid, fen in ROOT_FENS.items()}


def classify(moves: Iterable[chess.Move], window: int = ROOT_WINDOW) -> str | None:
    """Opening id whose root position the move sequence reaches within `window` plies."""
    board = chess.Board()
    for ply, move in enumerate(moves):
        if ply >= window:
            break
        board.push(move)
        hit = _FEN_TO_ID.get(fen_norm(board))
        if hit:
            return hit
    return None
