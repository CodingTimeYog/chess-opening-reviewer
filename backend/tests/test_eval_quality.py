import chess
import pytest
from helpers import RUY_A, RUY_C, game_rows, positions_frame

from chess_trainer import config
from chess_trainer.agents.base import Agent
from chess_trainer.agents.book import BookAgent
from chess_trainer.data.openings import fen_norm
from chess_trainer.engine import EvalCache, evaluate_positions
from chess_trainer.eval.quality import (
    agreement,
    clipped_cp,
    engine_quality,
    quality_markdown,
    sample_positions,
    top_k_hit,
    top_move,
)

START = fen_norm(chess.Board())


class FixedAgent(Agent):
    name = "fixed"

    def __init__(self, policy):
        super().__init__()
        self._policy = policy

    def policy(self, board):
        return dict(self._policy)


@pytest.fixture
def positions():
    return positions_frame(game_rows("train1", RUY_A, "train"), game_rows("test1", RUY_C, "test"))


def test_top_k_hit_is_the_expected_hit_under_random_tie_breaking():
    policy = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert top_k_hit(policy, "a", 1) == 1
    assert top_k_hit(policy, "b", 1) == 0
    assert top_k_hit(policy, "b", 3) == 1
    assert top_k_hit(policy, "zz", 3) == 0
    uniform = dict.fromkeys("abcd", 0.25)
    assert top_k_hit(uniform, "c", 1) == 0.25
    assert top_k_hit(uniform, "c", 3) == 0.75
    assert top_k_hit({"a": 0.4, "b": 0.3, "c": 0.3}, "c", 2) == 0.5


def test_top_move_breaks_ties_with_a_seeded_draw():
    uniform = dict.fromkeys(["a2a3", "b2b3", "c2c3", "d2d3"], 0.25)
    assert top_move({"a2a3": 0.2, "b2b3": 0.8}, START, seed=0) == "b2b3"
    assert top_move(uniform, START, seed=1) == top_move(uniform, START, seed=1)
    assert len({top_move(uniform, START, seed=s) for s in range(40)}) > 1


def test_agreement_with_master_moves_all_and_strict(positions):
    result = agreement(BookAgent.from_positions(positions), positions)
    ruy = result["openings"]["ruy_lopez"]
    # Plies 0-4: the book's only move is the master move. Ply 5: book says a6, master Nf6.
    # Plies 6-9 are out of book: uniform over legal moves.
    board, legal = chess.Board(), []
    for ply, san in enumerate(RUY_C.split()):
        if ply >= 6:
            legal.append(board.legal_moves.count())
        board.push_san(san)
    top1_strict = sum(1 / n for n in legal) / 4
    top3_strict = sum(3 / n for n in legal) / 4
    assert ruy["strict"]["n"] == 4
    assert ruy["strict"]["top1"] == pytest.approx(100 * top1_strict, abs=0.01)
    assert ruy["strict"]["top3"] == pytest.approx(100 * top3_strict, abs=0.01)
    assert ruy["all"]["n"] == 10
    assert ruy["all"]["top1"] == pytest.approx(100 * (5 + 4 * top1_strict) / 10, abs=0.01)
    assert ruy["all"]["top3"] == pytest.approx(100 * (5 + 4 * top3_strict) / 10, abs=0.01)
    assert result["sanity"] == {
        "positions_checked": 10,
        "illegal_moves": 0,
        "bad_distributions": 0,
        "illegal_master_moves": 0,
    }


def test_sanity_check_counts_illegal_moves_and_bad_distributions(positions):
    sanity = agreement(FixedAgent({"e2e5": 0.7}), positions)["sanity"]
    assert sanity["illegal_moves"] == 10
    assert sanity["bad_distributions"] == 10


def _many_positions():
    lines = ["e4 e5 Nf3 Nc6 Bb5 a6", "e4 e5 Nf3 Nc6 Bb5 Nf6", "e4 e5 Nf3 Nc6 Bb5 d6 d4 Bd7"]
    return positions_frame(
        game_rows("tr", lines[0], "train"),
        game_rows("te1", lines[1], "test"),
        game_rows("te2", lines[2], "test"),
        game_rows("va", lines[2], "val"),
    )


def test_sample_is_seeded_stratified_and_has_distinct_positions():
    frame = _many_positions()
    one = sample_positions(frame, n_per_stratum=2, seed=4)
    two = sample_positions(frame, n_per_stratum=2, seed=4)
    assert one.equals(two)
    assert one.groupby(["opening", "stratum"]).size().to_dict() == {
        ("ruy_lopez", "seen"): 2,
        ("ruy_lopez", "strict"): 2,
    }
    assert not one.duplicated(["opening", "fen_norm"]).any()
    assert set(one.columns) >= {"opening", "stratum", "fen_norm", "ply", "move_uci"}
    everything = sample_positions(frame, n_per_stratum=500, seed=4)
    # 6 distinct seen positions (plies 0-5) and 2 strict ones across the two test games
    assert everything.groupby("stratum").size().to_dict() == {"seen": 6, "strict": 2}


def test_clipped_cp_bounds_mates_and_huge_scores():
    assert clipped_cp(35, None) == 35
    assert clipped_cp(1500, None) == 1000
    assert clipped_cp(None, 3) == 1000
    assert clipped_cp(None, -2) == -1000


def _start_sample():
    frame = positions_frame(game_rows("tr", "d4", "train"), game_rows("te", "e4", "test"))
    return sample_positions(frame, n_per_stratum=5, seed=0)


def _start_evals(**overrides):
    evals = {m.uci(): (-50, None) for m in chess.Board().legal_moves}
    evals.update({"e2e4": (30, None), "d2d4": (20, None), "a2a3": (-80, None)})
    evals.update(overrides)
    return {START: evals}


def test_engine_quality_cpl_blunders_and_expected_cpl():
    sample, evals = _start_sample(), _start_evals()
    good = engine_quality(FixedAgent({"d2d4": 1.0}), sample, evals)["openings"]["ruy_lopez"]["seen"]
    assert good["n"] == 1
    assert good["top_move"]["mean_cpl"] == 10
    assert good["top_move"]["median_cpl"] == 10
    assert good["top_move"]["blunder_rate"] == 0
    assert good["top_move"]["mean_wdl_loss"] > 0
    assert good["master"]["mean_cpl"] == 0  # the master played e4, the engine's best
    assert good["master"]["mean_wdl_loss"] == 0
    assert good["fallback_rate"] == 0

    bad = engine_quality(FixedAgent({"a2a3": 1.0}), sample, evals)["openings"]["ruy_lopez"]["seen"]
    assert bad["top_move"]["mean_cpl"] == 110
    assert bad["top_move"]["blunder_rate"] == 100

    mixed = FixedAgent({"e2e4": 0.5, "a2a3": 0.5})
    out = engine_quality(mixed, sample, evals)["openings"]["ruy_lopez"]["seen"]
    assert out["policy"]["mean_cpl"] == 55
    assert out["policy"]["blunder_rate"] == 50


def test_engine_quality_handles_empty_strata_and_renders():
    result = engine_quality(FixedAgent({"d2d4": 1.0}), _start_sample(), _start_evals())
    assert result["openings"]["taimanov"]["strict"]["n"] == 0
    assert result["openings"]["taimanov"]["strict"]["top_move"]["mean_cpl"] is None
    agree = agreement(FixedAgent({"d2d4": 1.0}), positions_frame(game_rows("te", "e4", "test")))
    text = quality_markdown({"agreement": agree, "engine": result, "depth": 14, "sample_seed": 0})
    assert "Top-1" in text and "CPL" in text and "Ruy Lopez" in text


def test_eval_cache_round_trip(tmp_path):
    cache = EvalCache(tmp_path / "engine.sqlite")
    assert cache.get(START, 14) is None
    cache.put(START, 14, {"e2e4": (30, None), "a2a3": (None, -3)})
    assert cache.get(START, 14) == {"e2e4": (30, None), "a2a3": (None, -3)}
    assert cache.get(START, 10) is None
    cache.close()
    assert EvalCache(tmp_path / "engine.sqlite").get(START, 14) is not None


@pytest.mark.skipif(not config.STOCKFISH_PATH.exists(), reason="Stockfish not installed")
def test_stockfish_scores_every_legal_move_and_fills_the_cache(tmp_path):
    cache = EvalCache(tmp_path / "engine.sqlite")
    evals = evaluate_positions([START], depth=6, cache=cache, workers=1, progress=False)
    assert set(evals[START]) == {m.uci() for m in chess.Board().legal_moves}
    assert cache.get(START, 6) == evals[START]
    best = max(clipped_cp(*score) for score in evals[START].values())
    assert -100 < best < 150
