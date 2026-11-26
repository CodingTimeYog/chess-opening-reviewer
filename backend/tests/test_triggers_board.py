import chess
import pytest
from helpers import RUY_A, game_rows, positions_frame

from chess_trainer.agents.book import BookAgent
from chess_trainer.engine import MoveAnalysis
from chess_trainer.feedback import FeedbackBuilder, TriggerEngine
from chess_trainer.feedback.structure import board_facts
from chess_trainer.rag import Retriever

EXCHANGE = "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 d4 exd4 Qxd4 Qxd4 Nxd4"


class Quiet:
    def analyse(self, board, move):
        return MoveAnalysis(move.uci(), (0, None), (0, None))


@pytest.fixture
def builder():
    positions = positions_frame(game_rows("g", RUY_A, "train"))
    return FeedbackBuilder(BookAgent.from_positions(positions), {}, Retriever(positions), Quiet())


def play(builder, san, kinds):
    board, triggers, events = chess.Board(), TriggerEngine(), []
    for tok in san.split():
        move = board.parse_san(tok)
        events += triggers.on_move(builder.move_facts(board, move))
        board.push(move)
    return [e for e in events if e.type in kinds]


def facts_at(san: str) -> dict:
    """Board facts of the last move of a line."""
    board, tokens = chess.Board(), san.split()
    for tok in tokens[:-1]:
        board.push_san(tok)
    return board_facts(board, board.parse_san(tokens[-1]))


def test_board_facts_for_a_quiet_move_are_empty():
    assert facts_at("e4 e5 Nf3") == {
        "castled": None, "captured": None, "piece": "knight", "queens_off": False, "structure": [],
    }  # fmt: skip


def test_castling_fires_for_each_side_with_the_wing(builder):
    events = play(builder, "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Be7 Re1 O-O", {"castled"})
    assert [(e.ply, e.side, e.facts["move"]["castled"]) for e in events] == [
        (6, "white", "kingside"),
        (9, "black", "kingside"),
    ]
    assert events[0].feedback["text"] == "White castled kingside."
    assert events[0].severity == "info"
    assert facts_at("d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7 O-O-O")["castled"] == "queenside"


def test_minor_piece_trade_fires_when_the_recapture_completes_it(builder):
    events = play(builder, EXCHANGE, {"minor_piece_trade"})
    assert [(e.ply, e.side) for e in events] == [(7, "black")]
    assert events[0].feedback["text"] == "Minor pieces were traded: a knight for a bishop."
    # a lone capture of a minor piece is not a trade
    assert not play(builder, "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 Nf6", {"minor_piece_trade"})


def test_queens_off_fires_once_when_the_last_queen_leaves(builder):
    events = play(builder, EXCHANGE + " Bd7 Nc3", {"queens_off"})
    assert [(e.ply, e.side) for e in events] == [(12, "white")]
    assert events[0].feedback["text"] == "The queens are off the board."


def test_structure_shift_names_what_changed(builder):
    events = play(builder, EXCHANGE, {"structure_shift"})
    by_ply = {e.ply: e.facts["move"]["structure"] for e in events}
    assert by_ply[7] == ["Black has doubled pawns on the c-file"]  # ...dxc6
    assert "pawns were exchanged in the centre on d4" in by_ply[9]  # ...exd4
    assert "the d-file is open" in by_ply[10]  # Qxd4 removes the last d-pawn
    assert 6 not in by_ply  # Bxc6 changes no pawn
    text = next(e for e in events if e.ply == 7).feedback["text"]
    assert text == "The pawn structure changed: Black has doubled pawns on the c-file."


def test_isolated_pawns_and_a_locked_centre():
    assert facts_at("e4 e6 d4 d5 e5")["structure"] == ["the centre is locked on the e-file"]
    # 1.e4 d5 2.exd5 Qxd5 3.d4 e5 4.dxe5: White is left with no d- or e-pawn neighbours for...
    isolated = facts_at("d4 d5 c4 e6 Nc3 c5 cxd5 exd5 dxc5")["structure"]
    assert "Black has an isolated pawn on the d-file" in isolated
