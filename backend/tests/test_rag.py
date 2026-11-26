import chess
import numpy as np
import pytest
import torch
from helpers import RUY_A, RUY_C, board_after, game_rows, positions_frame

from chess_trainer.agents.model import PolicyNet
from chess_trainer.agents.policy import PolicyAgent
from chess_trainer.data.openings import ROOT_FENS, fen_norm
from chess_trainer.rag import RandomRetriever, Retriever, index_embeddings

ROOT = ROOT_FENS["ruy_lopez"]
TAIMANOV = "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7"


@pytest.fixture
def positions():
    return positions_frame(
        game_rows("strong", RUY_A, "train", elo=2750, year=2010),
        game_rows("recent", RUY_A, "train", elo=2600, year=2021),
        game_rows("old", RUY_A, "train", elo=2600, year=1999),
        game_rows("berlin", RUY_C, "train", elo=2500, year=2018, eco="C67"),
        game_rows("heldout", RUY_C, "test", elo=2800, year=2022, eco="C67"),
        game_rows("sicilian", TAIMANOV, "train", opening="taimanov", eco="B48"),
    )


def test_exact_hits_rank_by_average_rating_then_most_recent_year(positions):
    hits = Retriever(positions).retrieve(ROOT, k=5)
    assert [h.game_id for h in hits] == ["strong", "recent", "old", "berlin"]
    assert all(h.exact and h.score == 1.0 for h in hits)
    assert [h.move_uci for h in hits] == ["a7a6", "a7a6", "a7a6", "g8f6"]
    top = hits[0]
    assert (top.white, top.white_elo, top.year, top.eco, top.ply) == (
        "White strong",
        2750,
        2010,
        "C84",
        5,
    )
    assert top.fen_norm == ROOT
    assert top.to_dict()["game_id"] == "strong"


def test_index_is_train_only_by_default_and_all_games_for_the_app(positions):
    assert "heldout" not in {h.game_id for h in Retriever(positions).retrieve(ROOT, k=9)}
    everything = Retriever(positions, splits=None).retrieve(ROOT, k=9)
    assert everything[0].game_id == "heldout"  # 2800-rated, so it ranks first
    assert len(everything) == 5


def test_k_limits_hits_and_each_game_appears_once(positions):
    hits = Retriever(positions).retrieve(fen_norm(chess.Board()), k=3)
    assert len(hits) == 3 and len({h.game_id for h in hits}) == 3


def test_without_embeddings_an_unseen_position_returns_nothing(positions):
    unseen = fen_norm(board_after("e4 e5 Nf3 Nc6 Bb5 f5"))
    assert Retriever(positions).retrieve(unseen, k=5) == []


def _with_fake_embeddings(positions):
    """Embeds a position as a one-hot of its ply, so the nearest neighbours of a query are
    the indexed positions at the same ply."""
    retriever = Retriever(positions, embed=lambda fen, ply: np.eye(16)[ply])
    rows = positions[positions["split"] == "train"].drop_duplicates("fen_norm")
    retriever.set_embeddings(list(rows["fen_norm"]), np.eye(16)[rows["ply"].to_numpy()])
    return retriever


def test_embedding_neighbours_fill_up_when_exact_hits_run_short(positions):
    retriever = _with_fake_embeddings(positions)
    unseen = fen_norm(board_after("e4 e5 Nf3 Nc6 Bb5 f5"))  # ply 6, never played
    hits = retriever.retrieve(unseen, k=3, ply=6, opening="ruy_lopez")
    assert len(hits) == 3 and not any(h.exact for h in hits)
    assert all(h.ply == 6 and h.score == pytest.approx(1.0) for h in hits)
    # equal similarity: higher rating first, then the more recent game
    assert [h.game_id for h in hits] == ["strong", "recent", "old"]


def test_exact_hits_come_before_neighbours_and_games_are_not_repeated(positions):
    retriever = _with_fake_embeddings(positions)
    after_berlin = fen_norm(board_after("e4 e5 Nf3 Nc6 Bb5 Nf6"))  # only `berlin` reached it
    hits = retriever.retrieve(after_berlin, k=3, ply=6, opening="ruy_lopez")
    assert [(h.game_id, h.exact) for h in hits] == [
        ("berlin", True),
        ("strong", False),
        ("recent", False),
    ]


def test_neighbours_stay_within_the_requested_opening(positions):
    retriever = _with_fake_embeddings(positions)
    unseen = fen_norm(board_after("e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6"))  # ply 8
    hits = retriever.retrieve(unseen, k=5, ply=8, opening="taimanov")
    assert [h.game_id for h in hits] == ["sicilian"]


def test_random_baseline_is_seeded_same_opening_and_same_ply(positions):
    baseline = RandomRetriever(positions, seed=1)
    one = baseline.retrieve(ROOT, k=3, ply=5, opening="ruy_lopez")
    assert one == RandomRetriever(positions, seed=1).retrieve(ROOT, k=3, ply=5, opening="ruy_lopez")
    assert len(one) == 3 and len({h.game_id for h in one}) == 3
    assert all(h.opening == "ruy_lopez" and h.ply == 5 for h in one)
    assert "heldout" not in {h.game_id for h in one}
    assert all(h.exact for h in one)  # every Ruy game is at the root position at ply 5
    late = baseline.retrieve("somewhere new", k=9, ply=13, opening="ruy_lopez")
    assert {h.game_id for h in late} == {"strong", "recent", "old"}  # RUY_C ends at ply 9
    assert not any(h.exact for h in late)


def test_network_embeddings_in_batch_match_single_positions_and_index_the_corpus(positions):
    torch.manual_seed(0)
    agent = PolicyAgent(PolicyNet())
    boards = [board_after("e4 e5 Nf3"), board_after("e4 c5 Nf3 e6")]
    batch = agent.embed_batch([fen_norm(b) for b in boards], [b.ply() for b in boards])
    assert batch.shape == (2, 64)
    for row, board in zip(batch, boards, strict=True):
        assert np.allclose(row, agent.embed(board), atol=1e-5)

    retriever = Retriever(positions)
    index_embeddings(retriever, agent)
    unseen = board_after("e4 e5 Nf3 Nc6 Bb5 f5")
    hits = retriever.retrieve(fen_norm(unseen), k=4, ply=unseen.ply(), opening="ruy_lopez")
    assert len(hits) == 4 and all(-1.0 <= h.score <= 1.0001 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
