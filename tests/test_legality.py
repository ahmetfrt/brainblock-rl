import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.constants import BOARD_SHAPE
from brainblock_rl.core.legality import is_legal_placement, place_piece, placement_cells


def test_placement_cells_are_absolute_xy_coordinates():
    assert placement_cells("T", 0, 2, 1) == ((2, 1), (3, 1), (4, 1), (3, 2))


def test_empty_board_accepts_inside_placement():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)

    assert is_legal_placement(board, "O", 0, 6, 3)


def test_placement_outside_board_is_illegal():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)

    assert not is_legal_placement(board, "I", 0, 5, 0)
    assert not is_legal_placement(board, "I", 1, 0, 2)


def test_overlap_is_illegal():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)
    board = place_piece(board, "O", 0, 0, 0)

    assert not is_legal_placement(board, "T", 0, 0, 0)


def test_place_piece_writes_piece_value_to_board_y_x():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)
    placed = place_piece(board, "T", 0, 2, 1)

    assert np.count_nonzero(placed) == 4
    assert placed[1, 2] == 5
    assert placed[1, 3] == 5
    assert placed[1, 4] == 5
    assert placed[2, 3] == 5
    assert np.count_nonzero(board) == 0


def test_place_piece_rejects_illegal_placement():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)

    with pytest.raises(ValueError):
        place_piece(board, "I", 0, 6, 0)
