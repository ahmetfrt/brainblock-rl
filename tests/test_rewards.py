import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.constants import BOARD_SHAPE
from brainblock_rl.core.rewards import (
    SHAPED_BAD_EMPTY_REGION_PENALTY,
    SHAPED_LEGAL_REWARD,
    SHAPED_NO_LEGAL_ACTION_PENALTY,
    SHAPED_V2_BAD_EMPTY_REGION_PENALTY,
    SHAPED_V2_COMPLETION_REWARD,
    SHAPED_V2_INVALID_ACTION_REWARD,
    SHAPED_V2_NO_LEGAL_ACTION_PENALTY,
    SPARSE_COMPLETION_REWARD,
    SPARSE_INVALID_ACTION_REWARD,
    SPARSE_LEGAL_REWARD,
    TERMINAL_COMPLETION_REWARD,
    TERMINAL_INVALID_ACTION_REWARD,
    bad_empty_region_count,
    compute_step_reward,
    covered_area,
    reward_for_invalid_action,
    reward_for_placement,
    validate_reward_mode,
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


def test_sparse_reward_focuses_on_completion():
    assert reward_for_placement(reward_mode="sparse") == SPARSE_LEGAL_REWARD
    assert reward_for_placement(reward_mode="sparse", terminated=True) == SPARSE_COMPLETION_REWARD
    assert reward_for_invalid_action("sparse") == SPARSE_INVALID_ACTION_REWARD


def test_terminal_reward_only_scores_terminal_outcomes():
    assert reward_for_placement(reward_mode="terminal") == 0.0
    assert reward_for_placement(reward_mode="terminal", terminated=True) == TERMINAL_COMPLETION_REWARD
    assert reward_for_invalid_action("terminal") == TERMINAL_INVALID_ACTION_REWARD


def test_shaped_reward_penalizes_bad_empty_regions_and_no_continuation():
    board = np.ones(BOARD_SHAPE, dtype=np.int8)
    board[0, 0] = 0

    assert bad_empty_region_count(board) == 1
    assert reward_for_placement(
        reward_mode="shaped",
        board=board,
        next_legal_action_count=0,
    ) == (
        SHAPED_LEGAL_REWARD
        + SHAPED_BAD_EMPTY_REGION_PENALTY
        + SHAPED_NO_LEGAL_ACTION_PENALTY
    )


def test_shaped_v2_has_no_legal_placement_reward():
    board = np.ones(BOARD_SHAPE, dtype=np.int8)
    board[0, 0] = 0

    assert reward_for_placement(reward_mode="shaped_v2") == 0.0
    assert reward_for_placement(reward_mode="shaped_v2", terminated=True) == SHAPED_V2_COMPLETION_REWARD
    assert reward_for_invalid_action("shaped_v2") == SHAPED_V2_INVALID_ACTION_REWARD
    assert reward_for_placement(
        reward_mode="shaped_v2",
        board=board,
        next_legal_action_count=0,
    ) == (
        SHAPED_V2_BAD_EMPTY_REGION_PENALTY
        + SHAPED_V2_NO_LEGAL_ACTION_PENALTY
    )


def test_unknown_reward_mode_is_rejected():
    with pytest.raises(ValueError):
        validate_reward_mode("mystery")
