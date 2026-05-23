"""Placement legality checks for BrainBlock boards."""

from __future__ import annotations

import numpy as np

from brainblock_rl.core.constants import BOARD_HEIGHT, BOARD_SHAPE, BOARD_WIDTH, EMPTY_CELL
from brainblock_rl.core.pieces import Coordinate, get_piece_cells, piece_index


def validate_board(board: np.ndarray) -> np.ndarray:
    """Return ``board`` as an array after validating its ``board[y, x]`` shape."""

    array = np.asarray(board)
    if array.shape != BOARD_SHAPE:
        raise ValueError(f"Board must have shape {BOARD_SHAPE}, got {array.shape}")
    return array


def placement_cells(
    piece: str | int | np.integer,
    orientation: int,
    x: int,
    y: int,
) -> tuple[Coordinate, ...]:
    """Return absolute board cells occupied by a placement."""

    return tuple((x + dx, y + dy) for dx, dy in get_piece_cells(piece, orientation))


def cells_inside_board(cells: tuple[Coordinate, ...]) -> bool:
    """Return whether every cell is inside the 8 by 5 board."""

    return all(0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT for x, y in cells)


def cells_overlap(board: np.ndarray, cells: tuple[Coordinate, ...]) -> bool:
    """Return whether any absolute cell overlaps a filled board cell."""

    array = validate_board(board)
    return any(array[y, x] != EMPTY_CELL for x, y in cells)


def is_legal_placement(
    board: np.ndarray,
    piece: str | int | np.integer,
    orientation: int,
    x: int,
    y: int,
) -> bool:
    """Return whether a piece can be placed without leaving or overlapping the board."""

    validate_board(board)
    try:
        cells = placement_cells(piece, orientation, x, y)
    except (KeyError, ValueError):
        return False

    return cells_inside_board(cells) and not cells_overlap(board, cells)


def place_piece(
    board: np.ndarray,
    piece: str | int | np.integer,
    orientation: int,
    x: int,
    y: int,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Place a legal piece and return the resulting board.

    Filled cells store ``piece_index + 1`` so ``0`` remains the empty-cell value.
    """

    array = validate_board(board)
    cells = placement_cells(piece, orientation, x, y)
    if not cells_inside_board(cells) or cells_overlap(array, cells):
        raise ValueError("Cannot place piece: placement is illegal")

    target = array.copy() if copy else array
    value = piece_index(piece) + 1
    for cell_x, cell_y in cells:
        target[cell_y, cell_x] = value
    return target


__all__ = [
    "cells_inside_board",
    "cells_overlap",
    "is_legal_placement",
    "place_piece",
    "placement_cells",
    "validate_board",
]
