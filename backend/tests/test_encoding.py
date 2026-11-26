import chess
import numpy as np
import pytest
import torch

from chess_trainer.agents.encoding import (
    N_FROM_TO,
    N_PLANES,
    N_PROMO,
    encode_bitboards,
    encode_board,
    indices_to_move,
    legal_indices,
    move_to_indices,
    planes_from_bitboards,
)
from chess_trainer.agents.model import PolicyNet, legal_logits
from chess_trainer.data.openings import board_from, fen_norm

OWN_PAWN, OWN_KING, OPP_PAWN, OPP_KING = 0, 5, 6, 11
SIDE, OWN_OO, OWN_OOO, OPP_OO, OPP_OOO, EP, PLY = 12, 13, 14, 15, 16, 17, 18

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1b1kb1r/1pqp1ppp/p1n1pn2/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 2 8",
    "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # en passant
    "4k3/P6P/8/8/8/8/p6p/R3K2R w KQ - 0 40",  # promotions and castling, White
    "r3k2r/P6P/8/8/8/8/p6p/4K3 b kq - 0 40",  # promotions and castling, Black
    "1n2k3/P7/8/8/8/8/7p/4K1N1 b - - 0 40",  # capture promotions
]


def test_board_from_restores_the_ply():
    board = board_from(fen_norm(chess.Board(POSITIONS[1])), ply=5)
    assert board.ply() == 5 and board.turn == chess.BLACK
    assert board_from(fen_norm(chess.Board()), ply=0).ply() == 0


def test_start_position_has_19_planes_with_the_documented_layout():
    planes = encode_board(chess.Board())
    assert planes.shape == (N_PLANES, 8, 8) == (19, 8, 8)
    assert planes.dtype == np.float32
    assert planes[OWN_PAWN, 1].sum() == 8 and planes[OWN_PAWN].sum() == 8
    assert planes[OPP_PAWN, 6].sum() == 8 and planes[OPP_PAWN].sum() == 8
    assert planes[OWN_KING, 0, 4] == 1 and planes[OPP_KING, 7, 4] == 1
    assert planes[:12].sum() == 32
    assert (planes[SIDE] == 1).all()
    for plane in (OWN_OO, OWN_OOO, OPP_OO, OPP_OOO):
        assert (planes[plane] == 1).all()
    assert (planes[EP] == 0).all()
    assert (planes[PLY] == 0).all()


def test_black_to_move_is_seen_from_blacks_side():
    board = chess.Board()
    board.push_san("e4")
    planes = encode_board(board)
    # Black's own pawns sit on the second rank of the oriented board.
    assert planes[OWN_PAWN, 1].sum() == 8
    assert planes[OWN_KING, 0, 4] == 1
    # White's e4 pawn is an opponent pawn on e5 after mirroring.
    assert planes[OPP_PAWN, 4, 4] == 1 and planes[OPP_PAWN, 3, 4] == 0
    assert (planes[SIDE] == 0).all()
    assert (planes[EP] == 0).all()  # no black pawn can capture en passant
    assert planes[PLY, 0, 0] == pytest.approx(1 / 40)


def test_castling_and_en_passant_planes():
    lost_kingside = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK1R1 b Qkq - 0 1")
    planes = encode_board(lost_kingside)  # Black to move: own = black, opponent = white
    assert (planes[OWN_OO] == 1).all() and (planes[OWN_OOO] == 1).all()
    assert (planes[OPP_OO] == 0).all() and (planes[OPP_OOO] == 1).all()
    ep = encode_board(chess.Board(POSITIONS[3]))
    assert (ep[EP, :, 3] == 1).all() and ep[EP].sum() == 8


def test_ply_plane_is_normalised_and_capped():
    assert encode_board(board_from(fen_norm(chess.Board()), 0), ply=20)[PLY, 3, 3] == 0.5
    assert encode_board(chess.Board(), ply=90)[PLY, 0, 0] == 1.0


@pytest.mark.parametrize("fen", POSITIONS)
def test_a_position_and_its_colour_mirror_encode_identically_except_side_plane(fen):
    board = chess.Board(fen)
    mirrored = board.mirror()
    a, b = encode_board(board, ply=6), encode_board(mirrored, ply=6)
    keep = [i for i in range(N_PLANES) if i != SIDE]
    assert np.array_equal(a[keep], b[keep])
    assert a[SIDE, 0, 0] != b[SIDE, 0, 0]


def test_bitboards_round_trip_to_planes_in_batches():
    boards = [chess.Board(fen) for fen in POSITIONS]
    packed = np.stack([encode_bitboards(b) for b in boards])
    assert packed.shape == (len(boards), 18) and packed.dtype == np.uint64
    planes = planes_from_bitboards(packed, np.array([b.ply() for b in boards]))
    for i, board in enumerate(boards):
        assert np.array_equal(planes[i], encode_board(board))


@pytest.mark.parametrize("fen", POSITIONS)
def test_move_indices_round_trip_for_every_legal_move(fen):
    board = chess.Board(fen)
    seen = set()
    for move in board.legal_moves:
        from_to, promo = move_to_indices(move, board.turn)
        assert 0 <= from_to < N_FROM_TO and -1 <= promo < N_PROMO
        assert indices_to_move(from_to, promo, board.turn) == move
        seen.add((from_to, promo))
    assert len(seen) == board.legal_moves.count()  # no two legal moves share an index pair


def test_black_moves_are_mirrored_onto_whites_indices():
    white = move_to_indices(chess.Move.from_uci("e2e4"), chess.WHITE)
    black = move_to_indices(chess.Move.from_uci("e7e5"), chess.BLACK)
    assert white == black == (chess.E4 * 64 + chess.E2, -1)
    assert move_to_indices(chess.Move.from_uci("g8f6"), chess.BLACK) == move_to_indices(
        chess.Move.from_uci("g1f3"), chess.WHITE
    )
    # the same mirrored position gives the same index set for both colours
    board = chess.Board(POSITIONS[2])
    assert sorted(legal_indices(board)[1]) == sorted(legal_indices(board.mirror())[1])


def test_promotion_logit_indexing():
    # index = piece (N, B, R, Q) * 8 + target file, in the mover's orientation
    assert move_to_indices(chess.Move.from_uci("a7a8q"), chess.WHITE)[1] == 3 * 8 + 0
    assert move_to_indices(chess.Move.from_uci("a7a8n"), chess.WHITE)[1] == 0
    assert move_to_indices(chess.Move.from_uci("h7h8r"), chess.WHITE)[1] == 2 * 8 + 7
    assert move_to_indices(chess.Move.from_uci("a7b8b"), chess.WHITE)[1] == 1 * 8 + 1
    # Black's promotion on e1 is the mirror of White's on e8
    assert move_to_indices(chess.Move.from_uci("e2e1q"), chess.BLACK) == move_to_indices(
        chess.Move.from_uci("e7e8q"), chess.WHITE
    )
    # the four promotions of one pawn move share a from-to index and differ in promo index
    pairs = [move_to_indices(chess.Move.from_uci(f"a7a8{p}"), chess.WHITE) for p in "nbrq"]
    assert len({ft for ft, _ in pairs}) == 1 and len({p for _, p in pairs}) == 4


def test_model_shapes_and_parameter_count():
    torch.manual_seed(0)
    net = PolicyNet()
    assert sum(p.numel() for p in net.parameters()) == 465_441
    out = net(torch.zeros(3, N_PLANES, 8, 8))
    assert out.from_to.shape == (3, N_FROM_TO)
    assert out.promo.shape == (3, N_PROMO)
    assert out.value.shape == (3,) and out.value.abs().max() <= 1
    assert out.features.shape == (3, 64)
    no_value = PolicyNet(value_head=False)
    assert no_value(torch.zeros(1, N_PLANES, 8, 8)).value is None


def test_from_to_logit_layout_matches_the_move_index():
    # the logit for (from, to) must be read from plane `to`, cell `from`
    net = PolicyNet().eval()
    x = torch.randn(1, N_PLANES, 8, 8)
    with torch.no_grad():
        hidden = net.body(net.stem(x))
        planes = net.from_to(hidden)  # (1, 64 to-planes, 8, 8)
        flat = net(x).from_to
    move = chess.Move.from_uci("g1f3")
    index, _ = move_to_indices(move, chess.WHITE)
    rank, file = divmod(move.from_square, 8)
    assert flat[0, index] == planes[0, move.to_square, rank, file]


def test_legal_logits_mask_padding_and_add_promotion_terms():
    board = chess.Board(POSITIONS[4])
    moves, from_to, promo = legal_indices(board)
    width = len(moves) + 3
    legal_ft = torch.full((1, width), -1, dtype=torch.long)
    legal_promo = torch.full((1, width), -1, dtype=torch.long)
    legal_ft[0, : len(moves)] = torch.from_numpy(from_to)
    legal_promo[0, : len(moves)] = torch.from_numpy(promo)

    ft_logits = torch.zeros(1, N_FROM_TO)
    promo_logits = torch.zeros(1, N_PROMO)
    promo_logits[0, 3 * 8 + 0] = 5.0  # queen promotion on the a-file
    ft_logits[0, move_to_indices(chess.Move.from_uci("e1c1"), chess.WHITE)[0]] = 2.0
    logits = legal_logits(ft_logits, promo_logits, legal_ft, legal_promo)

    assert torch.isinf(logits[0, len(moves) :]).all()  # padding can never be chosen
    assert torch.isfinite(logits[0, : len(moves)]).all()
    by_move = {m.uci(): logits[0, i].item() for i, m in enumerate(moves)}
    assert by_move["a7a8q"] == 5.0 and by_move["a7a8r"] == 0.0 and by_move["h7h8q"] == 0.0
    assert by_move["e1c1"] == 2.0
    probs = torch.softmax(logits, dim=1)
    assert probs.sum().item() == pytest.approx(1.0)
    assert probs[0, len(moves) :].sum().item() == 0.0
