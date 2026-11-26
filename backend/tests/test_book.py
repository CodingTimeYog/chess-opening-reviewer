from collections import Counter

import chess
import pytest
from helpers import game_rows, positions_frame

from chess_trainer.agents.book import BookAgent


@pytest.fixture
def positions():
    return positions_frame(
        game_rows("g1", "e4 e5", "train", result="0-1"),
        game_rows("g2", "e4 c5", "train", result="1/2-1/2"),
        game_rows("g3", "e4 e6", "train", result="1-0"),
        game_rows("g4", "d4 d5", "train", result="1-0"),
        game_rows("g5", "a3 e5", "test"),
        game_rows("g6", "h3 e5", "val"),
    )


def test_policy_is_train_move_frequency(positions):
    book = BookAgent.from_positions(positions)
    board = chess.Board()
    assert book.policy(board) == {"e2e4": 0.75, "d2d4": 0.25}
    assert book.is_fallback(board) is False


def test_val_and_test_games_never_enter_the_book(positions):
    policy = BookAgent.from_positions(positions).policy(chess.Board())
    assert "a2a3" not in policy and "h2h3" not in policy


def test_out_of_book_is_uniform_over_legal_moves_and_flagged(positions):
    book = BookAgent.from_positions(positions)
    board = chess.Board()
    board.push_san("a3")
    policy = book.policy(board)
    assert book.is_fallback(board) is True
    assert set(policy) == {m.uci() for m in board.legal_moves}
    assert all(p == pytest.approx(1 / 20) for p in policy.values())


def test_result_weighting_favours_the_winners_moves(positions):
    book = BookAgent.from_positions(positions, result_weighted=True)
    # e4: lost (0.5) + drew (0.75) + won (1.0) = 2.25; d4: won = 1.0
    assert book.policy(chess.Board()) == pytest.approx({"e2e4": 2.25 / 3.25, "d2d4": 1 / 3.25})


def test_choose_is_seeded_and_samples_by_frequency(positions):
    board = chess.Board()
    one = BookAgent.from_positions(positions, seed=3)
    two = BookAgent.from_positions(positions, seed=3)
    assert [one.choose(board) for _ in range(20)] == [two.choose(board) for _ in range(20)]
    counts = Counter(one.choose(board).uci() for _ in range(2000))
    assert set(counts) == {"e2e4", "d2d4"}
    assert 0.70 < counts["e2e4"] / 2000 < 0.80


def test_temperature_zero_is_argmax(positions):
    book = BookAgent.from_positions(positions)
    assert {book.choose(chess.Board(), temperature=0).uci() for _ in range(20)} == {"e2e4"}
