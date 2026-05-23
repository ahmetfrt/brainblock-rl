from collections import Counter
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.constants import CELLS_PER_PIECE, ORIENTATION_COUNT, PIECE_TYPES
from brainblock_rl.core.pieces import create_initial_queue, get_piece_cells


def test_all_piece_orientations_have_four_unique_cells():
    for piece in PIECE_TYPES:
        for orientation in range(ORIENTATION_COUNT):
            cells = get_piece_cells(piece, orientation)

            assert len(cells) == CELLS_PER_PIECE
            assert len(set(cells)) == CELLS_PER_PIECE
            assert min(x for x, _ in cells) == 0
            assert min(y for _, y in cells) == 0


def test_i_piece_rotation_becomes_vertical():
    assert get_piece_cells("I", 0) == ((0, 0), (1, 0), (2, 0), (3, 0))
    assert get_piece_cells("I", 1) == ((0, 0), (0, 1), (0, 2), (0, 3))


def test_symmetric_pieces_keep_all_orientation_indices():
    orientations = [get_piece_cells("O", orientation) for orientation in range(ORIENTATION_COUNT)]

    assert len(orientations) == ORIENTATION_COUNT
    assert all(cells == ((0, 0), (1, 0), (0, 1), (1, 1)) for cells in orientations)


def test_initial_queue_has_two_of_each_piece_and_uses_rng():
    queue_a = create_initial_queue(np.random.default_rng(123))
    queue_b = create_initial_queue(np.random.default_rng(123))

    assert queue_a == queue_b
    assert len(queue_a) == 10
    assert Counter(queue_a) == {piece: 2 for piece in PIECE_TYPES}
