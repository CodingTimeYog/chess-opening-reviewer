import pytest
from helpers import RUY_A, RUY_C, game_rows, positions_frame

from chess_trainer.agents.book import BookAgent
from chess_trainer.eval.coverage import evaluate_coverage, metrics_markdown


@pytest.fixture
def positions():
    return positions_frame(
        game_rows("train1", RUY_A, "train"),
        game_rows("test1", RUY_C, "test"),
        game_rows("val1", RUY_A, "val"),
        game_rows("tai_train", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6", "train", opening="taimanov"),
        game_rows("tai_test", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6", "test", opening="taimanov"),
    )


def run(positions, tree=None, **kwargs):
    return evaluate_coverage(BookAgent.from_positions(positions), positions, tree or {}, **kwargs)


def test_position_coverage_all_vs_strict_per_opening(positions):
    ruy = run(positions)["openings"]["ruy_lopez"]["positions"]
    # RUY_C shares plies 0-5 with the train game, then leaves it for 4 positions.
    # Ply 0 (the start position) is also in the Taimanov train game.
    assert ruy["all"] == {"n": 10, "covered": 6, "pct": 60.0}
    assert ruy["strict"] == {"n": 4, "covered": 0, "pct": 0.0}
    tai = run(positions)["openings"]["taimanov"]["positions"]
    assert tai["all"] == {"n": 8, "covered": 8, "pct": 100.0}
    assert tai["strict"] == {"n": 0, "covered": 0, "pct": None}


def test_only_the_requested_split_is_scored(positions):
    val = run(positions, split="val")["openings"]["ruy_lopez"]["positions"]["all"]
    assert val == {"n": 14, "covered": 14, "pct": 100.0}


def test_line_coverage_counts_consecutive_known_own_moves(positions):
    lines = run(positions, min_own_moves=3)["openings"]["ruy_lopez"]["lines"]
    # White (repertoire side) is in book at plies 0, 2, 4 and out at ply 6.
    assert lines["repertoire_side"] == {"n": 1, "covered": 1, "pct": 100.0, "mean_depth": 3.0}
    assert lines["opponent_side"] == {"n": 1, "covered": 1, "pct": 100.0, "mean_depth": 3.0}
    lines = run(positions, min_own_moves=4)["openings"]["ruy_lopez"]["lines"]
    assert lines["repertoire_side"]["covered"] == 0
    assert lines["repertoire_side"]["n"] == 1


def test_lines_too_short_to_judge_are_left_out_of_the_denominator(positions):
    lines = run(positions, min_own_moves=6)["openings"]["ruy_lopez"]["lines"]
    assert lines["repertoire_side"] == {"n": 0, "covered": 0, "pct": None, "mean_depth": None}


def test_taimanov_repertoire_side_is_black():
    # The book knows only the start position, so White gets one covered move, Black none.
    frame = positions_frame(
        game_rows("t", "e4", "train", opening="taimanov"),
        game_rows("x", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6", "test", opening="taimanov"),
    )
    lines = run(frame, min_own_moves=1)["openings"]["taimanov"]["lines"]
    assert lines["opponent_side"]["mean_depth"] == 1.0
    assert lines["repertoire_side"]["mean_depth"] == 0.0


def test_prep_coverage_uses_in_repertoire_moves_for_the_repertoire_side(positions):
    start = positions.loc[positions["ply"] == 0, "fen_norm"].iloc[0]
    tree = {start: {"e2e4": {"in_repertoire": True}, "d2d4": {"in_repertoire": False}}}
    prep = run(positions, tree)["openings"]["ruy_lopez"]["prep"]
    # White is to move in 5 of the 10 test positions; the tree covers the first.
    assert prep == {"n": 5, "covered": 1, "pct": 20.0}


def test_metadata_and_markdown(positions):
    metrics = run(positions, seed=5, git_sha="abc1234")
    assert metrics["agent"] == "book"
    assert metrics["split"] == "test"
    assert metrics["seed"] == 5 and metrics["git_sha"] == "abc1234"
    text = metrics_markdown(metrics)
    assert "Ruy Lopez" in text and "Sicilian Taimanov" in text
    assert "60.0%" in text and "strict" in text
