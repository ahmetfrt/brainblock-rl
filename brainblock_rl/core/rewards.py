"""Reward helpers for BrainBlock environment variants."""

from __future__ import annotations

import numpy as np

from brainblock_rl.core.constants import BOARD_HEIGHT, BOARD_WIDTH, CELLS_PER_PIECE, EMPTY_CELL

LEGAL_CELL_REWARD = 1.0
INVALID_ACTION_REWARD = -1.0
COMPLETION_BONUS = 0.0

SPARSE_LEGAL_REWARD = 0.1
SPARSE_INVALID_ACTION_REWARD = -5.0
SPARSE_COMPLETION_REWARD = 10.0

TERMINAL_INVALID_ACTION_REWARD = -5.0
TERMINAL_COMPLETION_REWARD = 20.0

SHAPED_LEGAL_REWARD = 0.5
SHAPED_INVALID_ACTION_REWARD = -5.0
SHAPED_COMPLETION_BONUS = 10.0
SHAPED_BAD_EMPTY_REGION_PENALTY = -1.0
SHAPED_NO_LEGAL_ACTION_PENALTY = -0.5

SHAPED_V2_INVALID_ACTION_REWARD = -3.0
SHAPED_V2_COMPLETION_REWARD = 20.0
SHAPED_V2_BAD_EMPTY_REGION_PENALTY = -0.25
SHAPED_V2_NO_LEGAL_ACTION_PENALTY = -1.0
SHAPED_V2_REGION_PENALTY_MIN_COVERED_AREA = 20

REWARD_MODE_PLACEHOLDER = "placeholder"
REWARD_MODE_SPARSE = "sparse"
REWARD_MODE_SHAPED = "shaped"
REWARD_MODE_TERMINAL = "terminal"
REWARD_MODE_SHAPED_V2 = "shaped_v2"
VALID_REWARD_MODES = (
    REWARD_MODE_PLACEHOLDER,
    REWARD_MODE_SPARSE,
    REWARD_MODE_SHAPED,
    REWARD_MODE_TERMINAL,
    REWARD_MODE_SHAPED_V2,
)


def validate_reward_mode(reward_mode: str) -> str:
    """Return a valid reward mode or raise a clear error."""

    if reward_mode not in VALID_REWARD_MODES:
        choices = ", ".join(VALID_REWARD_MODES)
        raise ValueError(f"Unknown reward_mode {reward_mode!r}. Choices: {choices}")
    return reward_mode


def reward_for_placement(
    placed_cells: int = CELLS_PER_PIECE,
    *,
    terminated: bool = False,
    reward_mode: str = REWARD_MODE_PLACEHOLDER,
    board: np.ndarray | None = None,
    next_legal_action_count: int | None = None,
) -> float:
    """Return the reward for a legal placement."""

    reward_mode = validate_reward_mode(reward_mode)
    if placed_cells < 0:
        raise ValueError("placed_cells must be non-negative")

    if reward_mode == REWARD_MODE_SPARSE:
        return SPARSE_COMPLETION_REWARD if terminated else SPARSE_LEGAL_REWARD

    if reward_mode == REWARD_MODE_TERMINAL:
        return TERMINAL_COMPLETION_REWARD if terminated else 0.0

    if reward_mode == REWARD_MODE_SHAPED:
        reward = SHAPED_LEGAL_REWARD
        if board is not None:
            reward += bad_empty_region_count(board) * SHAPED_BAD_EMPTY_REGION_PENALTY
        if not terminated and next_legal_action_count == 0:
            reward += SHAPED_NO_LEGAL_ACTION_PENALTY
        if terminated:
            reward += SHAPED_COMPLETION_BONUS
        return float(reward)

    if reward_mode == REWARD_MODE_SHAPED_V2:
        if terminated:
            return SHAPED_V2_COMPLETION_REWARD

        reward = 0.0
        if board is not None and covered_area(board) >= SHAPED_V2_REGION_PENALTY_MIN_COVERED_AREA:
            reward += bad_empty_region_count(board) * SHAPED_V2_BAD_EMPTY_REGION_PENALTY
        if next_legal_action_count == 0:
            reward += SHAPED_V2_NO_LEGAL_ACTION_PENALTY
        return float(reward)

    bonus = COMPLETION_BONUS if terminated else 0.0
    return float(placed_cells * LEGAL_CELL_REWARD + bonus)


def reward_for_invalid_action(
    reward_mode: str = REWARD_MODE_PLACEHOLDER,
) -> float:
    """Return the hard-termination reward for an invalid action."""

    reward_mode = validate_reward_mode(reward_mode)
    if reward_mode == REWARD_MODE_SPARSE:
        return SPARSE_INVALID_ACTION_REWARD
    if reward_mode == REWARD_MODE_SHAPED:
        return SHAPED_INVALID_ACTION_REWARD
    if reward_mode == REWARD_MODE_TERMINAL:
        return TERMINAL_INVALID_ACTION_REWARD
    if reward_mode == REWARD_MODE_SHAPED_V2:
        return SHAPED_V2_INVALID_ACTION_REWARD
    return INVALID_ACTION_REWARD


def compute_step_reward(
    is_legal: bool,
    placed_cells: int = CELLS_PER_PIECE,
    *,
    terminated: bool = False,
    reward_mode: str = REWARD_MODE_PLACEHOLDER,
    board: np.ndarray | None = None,
    next_legal_action_count: int | None = None,
) -> float:
    """Return the reward for a step outcome."""

    if not is_legal:
        return reward_for_invalid_action(reward_mode)
    return reward_for_placement(
        placed_cells,
        terminated=terminated,
        reward_mode=reward_mode,
        board=board,
        next_legal_action_count=next_legal_action_count,
    )


def covered_area(board: np.ndarray) -> int:
    """Return the number of occupied board cells."""

    return int(np.count_nonzero(board))


def bad_empty_region_count(board: np.ndarray) -> int:
    """Count empty connected components whose size cannot be tiled by tetrominoes.

    Since every remaining piece occupies exactly four cells, an empty connected
    component whose size is not divisible by four is impossible to fill exactly.
    """

    array = np.asarray(board)
    empty = array == EMPTY_CELL
    visited = np.zeros_like(empty, dtype=bool)
    bad_regions = 0

    for start_y in range(BOARD_HEIGHT):
        for start_x in range(BOARD_WIDTH):
            if not empty[start_y, start_x] or visited[start_y, start_x]:
                continue

            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            component_size = 0
            while stack:
                x, y = stack.pop()
                component_size += 1
                for next_x, next_y in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if not (0 <= next_x < BOARD_WIDTH and 0 <= next_y < BOARD_HEIGHT):
                        continue
                    if visited[next_y, next_x] or not empty[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))

            if component_size % CELLS_PER_PIECE != 0:
                bad_regions += 1

    return bad_regions


__all__ = [
    "COMPLETION_BONUS",
    "INVALID_ACTION_REWARD",
    "LEGAL_CELL_REWARD",
    "REWARD_MODE_PLACEHOLDER",
    "REWARD_MODE_SHAPED",
    "REWARD_MODE_SHAPED_V2",
    "REWARD_MODE_SPARSE",
    "REWARD_MODE_TERMINAL",
    "SHAPED_BAD_EMPTY_REGION_PENALTY",
    "SHAPED_COMPLETION_BONUS",
    "SHAPED_INVALID_ACTION_REWARD",
    "SHAPED_LEGAL_REWARD",
    "SHAPED_NO_LEGAL_ACTION_PENALTY",
    "SHAPED_V2_BAD_EMPTY_REGION_PENALTY",
    "SHAPED_V2_COMPLETION_REWARD",
    "SHAPED_V2_INVALID_ACTION_REWARD",
    "SHAPED_V2_NO_LEGAL_ACTION_PENALTY",
    "SHAPED_V2_REGION_PENALTY_MIN_COVERED_AREA",
    "SPARSE_COMPLETION_REWARD",
    "SPARSE_INVALID_ACTION_REWARD",
    "SPARSE_LEGAL_REWARD",
    "TERMINAL_COMPLETION_REWARD",
    "TERMINAL_INVALID_ACTION_REWARD",
    "VALID_REWARD_MODES",
    "bad_empty_region_count",
    "compute_step_reward",
    "covered_area",
    "reward_for_invalid_action",
    "reward_for_placement",
    "validate_reward_mode",
]
