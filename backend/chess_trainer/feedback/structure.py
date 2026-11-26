"""Board-level facts about a move: castling, captures, queens leaving, pawn-structure changes."""

import chess

FILES = "abcdefgh"
CENTRE = {chess.D4, chess.D5, chess.E4, chess.E5}
COLOUR = {chess.WHITE: "White", chess.BLACK: "Black"}


def _signature(board: chess.Board) -> dict:
    files = {
        colour: [chess.square_file(sq) for sq in board.pieces(chess.PAWN, colour)]
        for colour in chess.COLORS
    }
    signature: dict = {
        "open": {
            f for f in range(8) if f not in files[chess.WHITE] and f not in files[chess.BLACK]
        },
        "locked": {
            chess.square_file(sq)
            for sq in board.pieces(chess.PAWN, chess.WHITE)
            if chess.square_file(sq) in (3, 4)
            and board.piece_at(sq + 8) == chess.Piece(chess.PAWN, chess.BLACK)
        },
    }
    for colour in chess.COLORS:
        mine = files[colour]
        signature[("doubled", colour)] = {f for f in set(mine) if mine.count(f) > 1}
        signature[("isolated", colour)] = {
            f for f in set(mine) if f - 1 not in mine and f + 1 not in mine
        }
    return signature


def _structure_changes(board: chess.Board, move: chess.Move, after: chess.Board) -> list[str]:
    before_sig, after_sig = _signature(board), _signature(after)
    changes = []
    captured = board.piece_at(move.to_square)
    pawn_takes_pawn = board.piece_type_at(move.from_square) == chess.PAWN and (
        board.is_en_passant(move) or (captured and captured.piece_type == chess.PAWN)
    )
    if pawn_takes_pawn and move.to_square in CENTRE:
        changes.append(f"pawns were exchanged in the centre on {chess.square_name(move.to_square)}")
    for f in sorted(after_sig["open"] - before_sig["open"]):
        changes.append(f"the {FILES[f]}-file is open")
    for colour in chess.COLORS:
        for f in sorted(after_sig[("doubled", colour)] - before_sig[("doubled", colour)]):
            changes.append(f"{COLOUR[colour]} has doubled pawns on the {FILES[f]}-file")
        for f in sorted(after_sig[("isolated", colour)] - before_sig[("isolated", colour)]):
            changes.append(f"{COLOUR[colour]} has an isolated pawn on the {FILES[f]}-file")
    for f in sorted(after_sig["locked"] - before_sig["locked"]):
        changes.append(f"the centre is locked on the {FILES[f]}-file")
    return changes


def board_facts(board: chess.Board, move: chess.Move) -> dict:
    after = board.copy(stack=False)
    after.push(move)
    castled = None
    if board.is_castling(move):
        castled = "kingside" if board.is_kingside_castling(move) else "queenside"
    captured = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    had_queens = bool(board.queens)
    return {
        "castled": castled,
        "captured": chess.piece_name(captured) if captured and not castled else None,
        "piece": chess.piece_name(board.piece_type_at(move.from_square)),
        "queens_off": had_queens and not after.queens,
        "structure": _structure_changes(board, move, after),
    }
