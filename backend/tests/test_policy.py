from collections import Counter

import chess
import numpy as np
import pytest
import torch
from helpers import RUY_A, RUY_C, game_rows, positions_frame

from chess_trainer.agents.dataset import batch_tensors, build_arrays
from chess_trainer.agents.encoding import encode_board, legal_indices
from chess_trainer.agents.model import PolicyNet
from chess_trainer.agents.policy import PolicyAgent
from chess_trainer.agents.train import train
from chess_trainer.data.openings import board_from

PROMOTIONS = "4k3/P6P/8/8/8/8/p6p/R3K2R w KQ - 0 40"


@pytest.fixture(scope="module")
def positions():
    return positions_frame(
        game_rows("g1", RUY_A, "train", result="1-0"),
        game_rows("g2", RUY_C, "train", result="0-1"),
        game_rows("g3", RUY_A, "train", result="1/2-1/2"),
        game_rows("g4", RUY_A, "val", result="1-0"),
    )


def test_arrays_share_repeated_positions_and_point_at_the_master_move(positions):
    train_rows = positions[positions["split"] == "train"]
    arrays = build_arrays(train_rows)
    assert len(arrays) == len(train_rows) == 38
    assert arrays.bitboards.shape[0] == train_rows["fen_norm"].nunique()
    for i, row in enumerate(train_rows.itertuples()):
        board = board_from(row.fen_norm, row.ply)
        moves, _, _ = legal_indices(board)
        assert moves[arrays.target[i]].uci() == row.move_uci
        assert arrays.ply[i] == row.ply


def test_result_weights_and_value_targets_are_from_the_movers_side(positions):
    rows = positions[positions["split"] == "train"].reset_index(drop=True)
    arrays = build_arrays(rows)
    g1, g2, g3 = (rows.index[rows["game_id"] == g] for g in ("g1", "g2", "g3"))
    # g1 is a White win: White's moves (even plies) weigh 1.0, Black's 0.5
    assert list(arrays.weight[g1][:2]) == [1.0, 0.5] and list(arrays.value[g1][:2]) == [1.0, -1.0]
    assert list(arrays.weight[g2][:2]) == [0.5, 1.0] and list(arrays.value[g2][:2]) == [-1.0, 1.0]
    assert set(arrays.weight[g3]) == {0.75} and set(arrays.value[g3]) == {0.0}
    unweighted = build_arrays(rows, result_weighted=False)
    assert set(unweighted.weight) == {1.0}


def test_batches_decode_to_the_same_planes_as_the_encoder(positions):
    rows = positions[positions["split"] == "train"].reset_index(drop=True)
    arrays = build_arrays(rows)
    batch = batch_tensors(arrays, np.array([0, 7, 20]))
    assert batch["x"].shape == (3, 19, 8, 8)
    for j, i in enumerate([0, 7, 20]):
        board = board_from(rows.loc[i, "fen_norm"], rows.loc[i, "ply"])
        assert np.array_equal(batch["x"][j].numpy(), encode_board(board))
        n_legal = board.legal_moves.count()
        assert (batch["legal_ft"][j, :n_legal] >= 0).all()
        assert (batch["legal_ft"][j, n_legal:] == -1).all()


def test_untrained_agent_gives_a_distribution_over_exactly_the_legal_moves():
    torch.manual_seed(0)
    agent = PolicyAgent(PolicyNet())
    for fen in (chess.STARTING_FEN, PROMOTIONS):
        board = chess.Board(fen)
        policy = agent.policy(board)
        assert set(policy) == {m.uci() for m in board.legal_moves}
        assert sum(policy.values()) == pytest.approx(1.0, abs=1e-5)
        assert min(policy.values()) >= 0
        assert agent.is_fallback(board) is False
    # under-promotions are separate moves with their own probabilities
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= set(agent.policy(chess.Board(PROMOTIONS)))
    assert agent.embed(chess.Board()).shape == (64,)


def test_agent_sees_black_positions_mirrored():
    torch.manual_seed(0)
    agent = PolicyAgent(PolicyNet())
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    policy, mirrored = agent.policy(board), agent.policy(board.mirror())
    # The two inputs differ only in the side-to-move plane, so the moves must line up
    # under mirroring even though the probabilities need not be identical.
    assert {chess.Move.from_uci(m).to_square ^ 56 for m in policy} == {
        chess.Move.from_uci(m).to_square for m in mirrored
    }


def test_play_samples_only_from_the_top_five_and_is_seeded():
    torch.manual_seed(0)
    one, two = PolicyAgent(PolicyNet(), seed=1), PolicyAgent(PolicyNet(), seed=1)
    two.model.load_state_dict(one.model.state_dict())
    board = chess.Board()
    picks = [one.choose(board).uci() for _ in range(200)]
    assert picks == [two.choose(board).uci() for _ in range(200)]
    policy = one.policy(board)
    top5 = set(sorted(policy, key=policy.get, reverse=True)[:5])
    assert set(Counter(picks)) <= top5 and len(set(picks)) > 1
    assert one.choose(board, temperature=0).uci() == max(sorted(policy), key=policy.get)


def test_training_learns_the_fixture_and_saves_the_best_checkpoint(positions, tmp_path):
    lines = []
    history = train(
        positions,
        name="tiny",
        out_dir=tmp_path,
        epochs=30,
        batch_size=8,
        lr=3e-3,
        patience=30,
        seed=0,
        threads=1,
        log=lines.append,
    )
    assert history["epochs"][-1]["train_loss"] < history["epochs"][0]["train_loss"]
    assert history["best_val_top1"] >= 85.0  # the val game repeats a train game
    assert {"epoch", "train_loss", "val_loss", "val_top1"} <= set(history["epochs"][0])
    assert any("val top-1" in line for line in lines)

    path = tmp_path / "tiny.pt"
    assert path.exists()
    agent = PolicyAgent.from_checkpoint(path)
    assert agent.best(chess.Board()) == "e2e4"
    saved = torch.load(path, weights_only=False)
    assert saved["val_top1"] == history["best_val_top1"]
    assert saved["config"] == {"blocks": 6, "channels": 64, "value_head": True}


def test_early_stopping_halts_when_val_top1_stops_improving(positions, tmp_path):
    history = train(
        positions, name="stop", out_dir=tmp_path, epochs=40, batch_size=8, lr=3e-3,
        patience=2, seed=0, threads=1, log=lambda _: None,
    )  # fmt: skip
    assert len(history["epochs"]) < 40
    assert history["stopped_early"] is True


def test_training_without_value_head(positions, tmp_path):
    history = train(
        positions, name="nov", out_dir=tmp_path, epochs=2, batch_size=8, value_head=False,
        seed=0, threads=1, log=lambda _: None,
    )  # fmt: skip
    assert PolicyAgent.from_checkpoint(tmp_path / "nov.pt").model.value is None
    assert len(history["epochs"]) == 2


class _Interrupt(Exception):
    pass


def _kill_after(epoch_line):
    def log(line):
        if line.startswith(epoch_line):
            raise _Interrupt

    return log


def _curve(history):
    return [(e["epoch"], e["train_loss"], e["val_loss"], e["val_top1"]) for e in history["epochs"]]


def test_resume_continues_exactly_where_an_interrupted_run_stopped(positions, tmp_path):
    kwargs = dict(epochs=4, batch_size=8, lr=3e-3, patience=10, seed=0, threads=1)
    straight = train(positions, name="a", out_dir=tmp_path, log=lambda _: None, **kwargs)

    with pytest.raises(_Interrupt):
        train(positions, name="b", out_dir=tmp_path, log=_kill_after("epoch 2/4"), **kwargs)
    last = torch.load(tmp_path / "b_last.pt", weights_only=False)
    assert last["epoch"] == 2
    assert {"model", "optimizer", "scheduler", "epoch", "best_val_top1"} <= set(last)

    lines = []
    resumed = train(positions, name="b", out_dir=tmp_path, resume=True, log=lines.append, **kwargs)
    assert any("resuming b from epoch 2" in line for line in lines)
    assert _curve(resumed) == _curve(straight)
    assert resumed["best_val_top1"] == straight["best_val_top1"]
    a = torch.load(tmp_path / "a_last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "b_last.pt", weights_only=False)["model"]
    assert all(torch.equal(a[key], b[key]) for key in a)
    # the best checkpoint is still the best epoch's weights, not the last epoch's
    best = torch.load(tmp_path / "b.pt", weights_only=False)
    assert best["val_top1"] == resumed["best_val_top1"] and best["epoch"] == resumed["best_epoch"]


def test_resume_without_a_checkpoint_starts_fresh_and_a_finished_run_is_not_retrained(
    positions, tmp_path
):
    kwargs = dict(epochs=2, batch_size=8, seed=0, threads=1)
    lines = []
    first = train(positions, name="c", out_dir=tmp_path, resume=True, log=lines.append, **kwargs)
    assert len(first["epochs"]) == 2
    assert not any("resuming" in line for line in lines)
    lines.clear()
    again = train(positions, name="c", out_dir=tmp_path, resume=True, log=lines.append, **kwargs)
    assert _curve(again) == _curve(first)
    assert any("already finished" in line for line in lines)


def test_resume_keeps_the_original_hyperparameters(positions, tmp_path):
    with pytest.raises(_Interrupt):
        train(
            positions, name="d", out_dir=tmp_path, epochs=3, batch_size=8, lr=3e-3, seed=0,
            threads=1, log=_kill_after("epoch 1/3"),
        )  # fmt: skip
    resumed = train(
        positions, name="d", out_dir=tmp_path, epochs=9, batch_size=64, lr=1.0, seed=5,
        threads=1, resume=True, log=lambda _: None,
    )  # fmt: skip
    assert resumed["hyperparameters"]["epochs"] == 3
    assert resumed["hyperparameters"]["lr"] == 3e-3
    assert len(resumed["epochs"]) == 3
