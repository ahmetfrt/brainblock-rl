import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.constants import BOARD_SHAPE
from brainblock_rl.core.rewards import (
    compute_step_reward,
    covered_area,
    reward_for_invalid_action,
    reward_for_placement,
)


def test_legal_placement_reward_counts_covered_cells():
    assert reward_for_placement() == 4.0
    assert compute_step_reward(True) == 4.0


def test_invalid_action_reward_is_negative():
    assert reward_for_invalid_action() == -1.0
    assert compute_step_reward(False) == -1.0


def test_covered_area_counts_nonzero_cells():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)
    board[0, 0] = 1
    board[4, 7] = 5

    assert covered_area(board) == 2
