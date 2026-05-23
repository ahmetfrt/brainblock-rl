"""Tetromino definitions and orientation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from brainblock_rl.core.constants import (
    CELLS_PER_PIECE,
    INDEX_TO_PIECE,
    ORIENTATION_COUNT,
    PIECES_PER_TYPE,
    PIECE_TO_INDEX,
    PIECE_TYPES,
)

Coordinate = tuple[int, int]
PieceCells = tuple[Coordinate, ...]

_BASE_CELLS: dict[str, PieceCells] = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "L": ((0, 0), (0, 1), (0, 2), (1, 2)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
}


def piece_name(piece: str | int | np.integer) -> str:
    """Return the canonical piece name for a name or piece index."""

    if isinstance(piece, (int, np.integer)):
        piece_index = int(piece)
        if piece_index not in INDEX_TO_PIECE:
            raise ValueError(f"Unknown piece index: {piece_index}")
        return INDEX_TO_PIECE[piece_index]

    normalized = str(piece).upper()
    if normalized not in PIECE_TO_INDEX:
        raise ValueError(f"Unknown piece type: {piece}")
    return normalized


def piece_index(piece: str | int | np.integer) -> int:
    """Return the stable integer index for a piece."""

    return PIECE_TO_INDEX[piece_name(piece)]


def _normalize(cells: Sequence[Coordinate]) -> PieceCells:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    normalized = ((x - min_x, y - min_y) for x, y in cells)
    return tuple(sorted(normalized, key=lambda cell: (cell[1], cell[0])))


def _rotate_clockwise(cells: Sequence[Coordinate]) -> list[Coordinate]:
    return [(y, -x) for x, y in cells]


def _reflect_horizontally(cells: Sequence[Coordinate]) -> list[Coordinate]:
    return [(-x, y) for x, y in cells]


def _orient_cells(cells: PieceCells, orientation: int) -> PieceCells:
    if not 0 <= orientation < ORIENTATION_COUNT:
        raise ValueError(f"Orientation must be in [0, 7], got {orientation}")

    transformed: list[Coordinate] = list(cells)
    if orientation >= 4:
        transformed = _reflect_horizontally(transformed)

    for _ in range(orientation % 4):
        transformed = _rotate_clockwise(transformed)

    normalized = _normalize(transformed)
    if len(set(normalized)) != CELLS_PER_PIECE:
        raise ValueError("Tetromino orientation must occupy four unique cells")
    return normalized


PIECE_ORIENTATIONS: dict[str, tuple[PieceCells, ...]] = {
    piece: tuple(_orient_cells(cells, orientation) for orientation in range(ORIENTATION_COUNT))
    for piece, cells in _BASE_CELLS.items()
}


def get_piece_cells(piece: str | int | np.integer, orientation: int) -> PieceCells:
    """Return normalized ``(x, y)`` cells for a piece orientation.

    The eight orientation indices represent the four rotations of the base
    piece, followed by the same four rotations after a horizontal reflection.
    Symmetric pieces intentionally keep duplicate orientations so the action
    space remains fixed.
    """

    return PIECE_ORIENTATIONS[piece_name(piece)][orientation]


def create_initial_queue(rng: np.random.Generator | None = None) -> list[str]:
    """Create and shuffle the 10-piece inventory using the provided RNG."""

    generator = rng if rng is not None else np.random.default_rng()
    queue = [piece for piece in PIECE_TYPES for _ in range(PIECES_PER_TYPE)]
    generator.shuffle(queue)
    return queue


__all__ = [
    "Coordinate",
    "PIECE_ORIENTATIONS",
    "PieceCells",
    "create_initial_queue",
    "get_piece_cells",
    "piece_index",
    "piece_name",
]
