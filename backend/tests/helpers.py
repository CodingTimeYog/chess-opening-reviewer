"""Shared test helpers."""

import chess
import pandas as pd

from chess_trainer.data.openings import fen_norm

RUY_A = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6"
RUY_C = "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6"


def game_rows(game_id: str, san: str, split: str, result: str = "1-0", opening: str = "ruy_lopez"):
    board, rows = chess.Board(), []
    for ply, tok in enumerate(san.split()):
        move = board.parse_san(tok)
        rows.append(
            {
                "game_id": game_id,
                "opening": opening,
                "split": split,
                "ply": ply,
                "fen_norm": fen_norm(board),
                "move_uci": move.uci(),
                "result": result,
            }
        )
        board.push(move)
    return rows


def positions_frame(*games: list[dict]) -> pd.DataFrame:
    from chess_trainer.data.split import mark_strict

    frame = pd.DataFrame([row for game in games for row in game])
    frame["strict"] = mark_strict(frame)
    return frame
