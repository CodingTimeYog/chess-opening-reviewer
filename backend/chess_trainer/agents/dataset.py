"""Position rows -> arrays for training. Repeated positions are encoded once."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from chess_trainer.agents.book import result_weights
from chess_trainer.agents.encoding import encode_bitboards, legal_indices, planes_from_bitboards
from chess_trainer.data.openings import board_from


@dataclass
class PositionArrays:
    # one entry per distinct position
    bitboards: np.ndarray  # (U, 18) uint64
    legal_ft: np.ndarray  # (U, L) int16, -1 padded
    legal_promo: np.ndarray  # (U, L) int8, -1 for none or padding
    # one entry per row
    position: np.ndarray  # (N,) index into the arrays above
    ply: np.ndarray  # (N,)
    target: np.ndarray  # (N,) index of the master move within the position's legal moves
    weight: np.ndarray  # (N,) loss weight
    value: np.ndarray  # (N,) game result for the side to move: 1, 0 or -1

    def __len__(self) -> int:
        return len(self.position)


def build_arrays(rows: pd.DataFrame, result_weighted: bool = True) -> PositionArrays:
    fens, position = np.unique(rows["fen_norm"].to_numpy(), return_inverse=True)
    encoded, move_index = [], []
    for fen in fens:
        board = board_from(fen)
        moves, from_to, promo = legal_indices(board)
        encoded.append((encode_bitboards(board), from_to, promo))
        move_index.append({move.uci(): i for i, move in enumerate(moves)})
    width = max(len(from_to) for _, from_to, _ in encoded)
    legal_ft = np.full((len(fens), width), -1, dtype=np.int16)
    legal_promo = np.full((len(fens), width), -1, dtype=np.int8)
    for i, (_, from_to, promo) in enumerate(encoded):
        legal_ft[i, : len(from_to)] = from_to
        legal_promo[i, : len(promo)] = promo

    target = [move_index[p][uci] for p, uci in zip(position, rows["move_uci"], strict=True)]
    weights = result_weights(rows) if result_weighted else pd.Series(1.0, index=rows.index)
    white_to_move = rows["ply"].to_numpy() % 2 == 0
    white_result = rows["result"].map({"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}).to_numpy()
    return PositionArrays(
        bitboards=np.stack([bitboards for bitboards, _, _ in encoded]),
        legal_ft=legal_ft,
        legal_promo=legal_promo,
        position=position.astype(np.int32),
        ply=rows["ply"].to_numpy().astype(np.int16),
        target=np.array(target, dtype=np.int64),
        weight=weights.to_numpy().astype(np.float32),
        value=np.where(white_to_move, white_result, -white_result).astype(np.float32),
    )


def batch_tensors(arrays: PositionArrays, index: np.ndarray) -> dict[str, torch.Tensor]:
    position = arrays.position[index]
    return {
        "x": torch.from_numpy(planes_from_bitboards(arrays.bitboards[position], arrays.ply[index])),
        "legal_ft": torch.from_numpy(arrays.legal_ft[position].astype(np.int64)),
        "legal_promo": torch.from_numpy(arrays.legal_promo[position].astype(np.int64)),
        "target": torch.from_numpy(arrays.target[index]),
        "weight": torch.from_numpy(arrays.weight[index]),
        "value": torch.from_numpy(arrays.value[index]),
    }
