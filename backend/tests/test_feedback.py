import chess
import pytest
from helpers import RUY_A, RUY_C, board_after, game_rows, positions_frame

from chess_trainer.agents.book import BookAgent
from chess_trainer.data.openings import fen_norm
from chess_trainer.engine import MoveAnalysis
from chess_trainer.feedback import FeedbackBuilder, TriggerEngine, classify_loss, render_move
from chess_trainer.rag import Retriever

AFTER_A6 = "e4 e5 Nf3 Nc6 Bb5 a6"


class StubAnalyser:
    """Says Ba4 is best after 3...a6 and scores moves from a table (default: 10 cp worse)."""

    def __init__(self, losses=None):
        self.losses = losses or {}

    def analyse(self, board, move):
        best = "b5a4" if fen_norm(board) == fen_norm(board_after(AFTER_A6)) else move.uci()
        loss = 0 if move.uci() == best else self.losses.get(move.uci(), 10)
        return MoveAnalysis(best_move=best, best=(40, None), played=(40 - loss, None))


@pytest.fixture
def positions():
    return positions_frame(
        game_rows("g1", RUY_A, "train", elo=2700, year=2012),
        game_rows("g2", RUY_A, "train", elo=2650, year=2019),
        game_rows("g3", RUY_C, "train", elo=2600, year=2005, eco="C67"),
        game_rows("g4", "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6", "train", elo=2550, eco="C68"),
    )


@pytest.fixture
def tree():
    return {
        fen_norm(board_after(AFTER_A6)): {
            "b5a4": {
                "san": "Ba4",
                "side": "own",
                "in_repertoire": True,
                "nags": [],
                "comment": "Keep the bishop; the pin can wait.",
            },
            "b5c6": {
                "san": "Bxc6",
                "side": "own",
                "in_repertoire": False,
                "nags": [6],
                "comment": "The Exchange gives up the bishop pair.",
            },
        }
    }


@pytest.fixture
def builder(positions, tree):
    return FeedbackBuilder(
        BookAgent.from_positions(positions),
        tree,
        Retriever(positions),
        StubAnalyser({"b5c6": 60, "b5e2": 130, "b5a6": 420}),
    )


def test_classification_thresholds():
    assert [classify_loss(x) for x in (0, 49, 50, 99, 100, 199, 200, 900)] == [
        "good", "good", "inaccuracy", "inaccuracy", "mistake", "mistake", "blunder", "blunder",
    ]  # fmt: skip


def test_facts_for_a_mistake_carry_engine_master_repertoire_and_citations(builder):
    board = board_after(AFTER_A6)
    feedback = builder.move_feedback(board, chess.Move.from_uci("b5e2"))
    facts = feedback["facts"]
    assert facts["played"] == {"uci": "b5e2", "san": "Be2"}
    assert facts["classification"] == "mistake" and facts["cp_loss"] == 130
    assert facts["best"] == {"uci": "b5a4", "san": "Ba4"}
    assert facts["ply"] == 6 and facts["side"] == "white"
    assert facts["master"]["in_book"] is True and facts["master"]["games"] == 3
    assert facts["master"]["played_games"] == 0
    assert facts["master"]["top"][0] == {"uci": "b5a4", "san": "Ba4", "games": 2, "share": 66.7}
    assert facts["repertoire"]["covered"] is True
    assert facts["repertoire"]["played_in_repertoire"] is False
    assert facts["repertoire"]["moves"] == [
        {"uci": "b5a4", "san": "Ba4", "comment": "Keep the bishop; the pin can wait."}
    ]
    assert facts["repertoire"]["played_note"] is None
    # citations: games that reached this exact position, best rated first, at most three
    assert [c["game_id"] for c in feedback["citations"]] == ["g1", "g2", "g4"]
    assert feedback["citations"][0]["move_san"] == "Ba4"
    assert feedback["citations"] == facts["citations"]


def test_template_text_states_every_fact(builder):
    board = board_after(AFTER_A6)
    text = builder.move_feedback(board, chess.Move.from_uci("b5e2"))["text"]
    assert text.startswith("Be2 is a mistake")
    assert "1.3 pawns" in text
    assert "Stockfish prefers Ba4" in text
    assert "Ba4 (2 of 3 games, 67%)" in text
    assert "Your repertoire move is Ba4" in text and "Keep the bishop" in text
    assert "White g1 - Black g1, Event g1 2012 (1-0), continued Ba4" in text
    assert text == render_move(builder.move_feedback(board, chess.Move.from_uci("b5e2"))["facts"])


def test_a_move_the_prep_discusses_and_rejects_gets_its_note(builder):
    feedback = builder.move_feedback(board_after(AFTER_A6), chess.Move.from_uci("b5c6"))
    assert feedback["facts"]["classification"] == "inaccuracy"
    assert feedback["facts"]["repertoire"]["played_note"] == {
        "nags": ["?!"],
        "comment": "The Exchange gives up the bishop pair.",
    }
    assert "Your prep marks Bxc6 as ?!" in feedback["text"]
    assert "The Exchange gives up the bishop pair." in feedback["text"]


def test_a_good_repertoire_move_and_a_position_outside_book_and_prep(builder):
    good = builder.move_feedback(board_after(AFTER_A6), chess.Move.from_uci("b5a4"))
    assert good["facts"]["classification"] == "good" and good["facts"]["cp_loss"] == 0
    assert good["text"].startswith("Ba4 is the engine's first choice")
    assert "your repertoire move" in good["text"]

    unknown = board_after("e4 e5 Nf3 Nc6 Bb5 f5 d3")
    feedback = builder.move_feedback(unknown, chess.Move.from_uci("g8f6"))
    assert feedback["facts"]["master"] == {
        "in_book": False,
        "games": 0,
        "top": [],
        "played_games": 0,
    }
    assert feedback["facts"]["repertoire"]["covered"] is False
    assert feedback["citations"] == []
    assert "No master game in the corpus reached this position" in feedback["text"]


def test_a_different_move_that_loses_nothing_is_not_called_the_first_choice(positions, tree):
    builder = FeedbackBuilder(
        BookAgent.from_positions(positions), tree, Retriever(positions), StubAnalyser({"d2d3": 0})
    )
    feedback = builder.move_feedback(board_after(AFTER_A6), chess.Move.from_uci("d2d3"))
    assert feedback["facts"]["cp_loss"] == 0 and feedback["facts"]["best"]["san"] == "Ba4"
    assert feedback["text"].startswith("d3 is as good as the engine's first choice, Ba4.")


def _play(builder, san, triggers, analyse=lambda ply: True):
    board, events = chess.Board(), []
    for ply, tok in enumerate(san.split()):
        move = board.parse_san(tok)
        facts = builder.move_feedback(board, move, analyse=analyse(ply))["facts"]
        events += triggers.on_move(facts)
        board.push(move)
    return events


def test_left_repertoire_fires_once_for_the_side_that_leaves(builder):
    events = _play(builder, "e4 e5 Nf3 Nc6 Bb5 a6 Be2 Nf6 d3", TriggerEngine())
    left = [e for e in events if e.type == "left_repertoire"]
    assert len(left) == 1
    event = left[0].to_dict()
    assert (event["ply"], event["severity"], event["side"]) == (6, "info", "white")
    assert "left your repertoire" in event["feedback"]["text"]
    assert "Ba4" in event["feedback"]["text"]
    assert set(event["feedback"]) == {"text", "facts", "citations"}


def test_left_book_fires_once_per_side_at_the_first_move_no_master_played(builder):
    events = _play(builder, "e4 e5 Nf3 Nc6 Bb5 a6 Be2 Nf6 d3", TriggerEngine())
    left = [e for e in events if e.type == "left_book"]
    # White leaves the book with Be2; afterwards nobody is in book, so Black cannot leave it.
    assert [(e.ply, e.side) for e in left] == [(6, "white")]
    assert "has not been played" in left[0].feedback["text"]
    assert "Ba4" in left[0].feedback["text"]

    black = _play(builder, "e4 e5 Nf3 Nc6 Bb5 Qe7", TriggerEngine())
    assert [(e.type, e.ply, e.side) for e in black if e.type == "left_book"] == [
        ("left_book", 5, "black")
    ]


def test_eval_swing_severity_follows_the_centipawn_loss(builder):
    events = _play(builder, "e4 e5 Nf3 Nc6 Bb5 a6 Bxa6", TriggerEngine())
    swings = [e for e in events if e.type == "eval_swing"]
    assert [(e.ply, e.severity) for e in swings] == [(6, "blunder")]
    assert swings[0].feedback["text"].startswith("Bxa6 is a blunder")
    quiet = _play(builder, "e4 e5 Nf3 Nc6 Bb5 a6 Ba4", TriggerEngine())
    assert not [e for e in quiet if e.type == "eval_swing"]
    # moves that were not analysed (the agent's own) never produce an eval swing
    skipped = _play(
        builder, "e4 e5 Nf3 Nc6 Bb5 a6 Bxa6", TriggerEngine(), analyse=lambda ply: False
    )
    assert not [e for e in skipped if e.type == "eval_swing"]


def test_opening_ends_after_three_agent_moves_outside_book_and_prep(builder):
    line = "e4 e5 Nf3 Nc6 Bb5 f5 d3 d6 Nc3 Nf6 O-O Be7 Re1 O-O"
    events = _play(builder, line, TriggerEngine(agent_color=chess.BLACK))
    ends = [e for e in events if e.type == "opening_end"]
    # 3...f5 is played from a book position; Black's moves at plies 7, 9, 11 are from unknown ones.
    assert [(e.ply, e.facts["reason"]) for e in ends] == [(11, "out_of_book")]
    assert "opening phase is over" in ends[0].feedback["text"]


def test_opening_ends_at_the_ply_cap_and_only_once(builder):
    triggers = TriggerEngine(max_ply=8)
    events = _play(builder, "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7", triggers)
    ends = [e for e in events if e.type == "opening_end"]
    assert [(e.ply, e.facts["reason"]) for e in ends] == [(7, "ply_cap")]
