"""Reward helpers for the initial BrainBlock environment."""

from __future__ import annotations

import numpy as np

from brainblock_rl.core.constants import CELLS_PER_PIECE

LEGAL_CELL_REWARD = 1.0
INVALID_ACTION_REWARD = -1.0
COMPLETION_BONUS = 0.0


def reward_for_placement(
    placed_cells: int = CELLS_PER_PIECE,
    *,
    terminated: bool = False,
) -> float:
    """Return the reward for a legal placement."""

    if placed_cells < 0:
        raise ValueError("placed_cells must be non-negative")
    bonus = COMPLETION_BONUS if terminated else 0.0
    return float(placed_cells * LEGAL_CELL_REWARD + bonus)


def reward_for_invalid_action() -> float:
    """Return the hard-termination reward for an invalid action."""

    return INVALID_ACTION_REWARD


def compute_step_reward(
    is_legal: bool,
    placed_cells: int = CELLS_PER_PIECE,
    *,
    terminated: bool = False,
) -> float:
    """Return the reward for a step outcome."""

    if not is_legal:
        return reward_for_invalid_action()
    return reward_for_placement(placed_cells, terminated=terminated)


def covered_area(board: np.ndarray) -> int:
    """Return the number of occupied board cells."""

    return int(np.count_nonzero(board))


__all__ = [
    "COMPLETION_BONUS",
    "INVALID_ACTION_REWARD",
    "LEGAL_CELL_REWARD",
    "compute_step_reward",
    "covered_area",
    "reward_for_invalid_action",
    "reward_for_placement",
]
