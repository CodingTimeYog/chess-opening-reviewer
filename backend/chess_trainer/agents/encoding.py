"""Board and move encoding for the learned policy.

The board is always seen from the side to move: with Black to move the position is
mirrored vertically and the colours swapped, so "own" pieces start on the first two ranks.

Input, 19 planes of 8x8, indexed [plane, rank, file] with rank 0 the mover's first rank:
  0-5   own P N B R Q K          6-11  opponent P N B R Q K
  12    side to move (1 = White)
  13-16 castling rights: own O-O, own O-O-O, opponent O-O, opponent O-O-O
  17    en-passant file (only when a capture is legal)
  18    ply / 40, capped at 1

Moves, in the same orientation:
  from-to index = to_square * 64 + from_square                      (4096)
  promotion index = piece (N, B, R, Q) * 8 + file of the to-square  (32), -1 if none
A move's logit is its from-to logit plus, for a promotion, its promotion logit.
"""

import chess
import numpy as np

from chess_trainer.config import MAX_PLY

N_PLANES = 19
N_BITBOARDS = 18  # every plane except ply, which is a scalar per position
N_FROM_TO = 64 * 64
N_PROMO = 4 * 8
PROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
_FULL = chess.BB_ALL


def encode_bitboards(board: chess.Board) -> np.ndarray:
    """The 18 binary planes as uint64 bitboards (bit = square in the mover's orientation)."""
    b = board if board.turn == chess.WHITE else board.mirror()
    planes = [b.pieces_mask(piece, chess.WHITE) for piece in chess.PIECE_TYPES]
    planes += [b.pieces_mask(piece, chess.BLACK) for piece in chess.PIECE_TYPES]
    planes.append(_FULL if board.turn == chess.WHITE else 0)
    planes.append(_FULL if b.has_kingside_castling_rights(chess.WHITE) else 0)
    planes.append(_FULL if b.has_queenside_castling_rights(chess.WHITE) else 0)
    planes.append(_FULL if b.has_kingside_castling_rights(chess.BLACK) else 0)
    planes.append(_FULL if b.has_queenside_castling_rights(chess.BLACK) else 0)
    en_passant = b.ep_square if b.has_legal_en_passant() else None
    planes.append(chess.BB_FILES[chess.square_file(en_passant)] if en_passant is not None else 0)
    return np.array(planes, dtype=np.uint64)


def planes_from_bitboards(bitboards: np.ndarray, ply: np.ndarray) -> np.ndarray:
    """(N, 18) uint64 and (N,) plies -> (N, 19, 8, 8) float32."""
    n = len(bitboards)
    as_bytes = bitboards.astype("<u8").view(np.uint8).reshape(n, N_BITBOARDS, 8)
    binary = np.unpackbits(as_bytes, axis=2, bitorder="little").reshape(n, N_BITBOARDS, 8, 8)
    out = np.empty((n, N_PLANES, 8, 8), dtype=np.float32)
    out[:, :N_BITBOARDS] = binary
    scaled = np.minimum(np.asarray(ply, dtype=np.float32), MAX_PLY) / MAX_PLY
    out[:, N_BITBOARDS] = scaled[:, None, None]
    return out


def encode_board(board: chess.Board, ply: int | None = None) -> np.ndarray:
    ply = board.ply() if ply is None else ply
    return planes_from_bitboards(encode_bitboards(board)[None], np.array([ply]))[0]


def _orient(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def move_to_indices(move: chess.Move, turn: chess.Color) -> tuple[int, int]:
    """(from-to index, promotion index or -1) of a move made by `turn`."""
    origin, target = _orient(move.from_square, turn), _orient(move.to_square, turn)
    promo = -1
    if move.promotion is not None:
        promo = PROMO_PIECES.index(move.promotion) * 8 + chess.square_file(target)
    return target * 64 + origin, promo


def indices_to_move(from_to: int, promo: int, turn: chess.Color) -> chess.Move:
    target, origin = divmod(int(from_to), 64)
    piece = None if promo < 0 else PROMO_PIECES[int(promo) // 8]
    return chess.Move(_orient(origin, turn), _orient(target, turn), promotion=piece)


def legal_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray, np.ndarray]:
    """Legal moves in UCI order with their from-to and promotion indices."""
    moves = sorted(board.legal_moves, key=chess.Move.uci)
    pairs = [move_to_indices(move, board.turn) for move in moves]
    from_to = np.array([p[0] for p in pairs], dtype=np.int64)
    promo = np.array([p[1] for p in pairs], dtype=np.int64)
    return moves, from_to, promo
