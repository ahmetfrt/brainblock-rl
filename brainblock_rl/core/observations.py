"""Observation helpers for the BrainBlock Gymnasium environment."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from brainblock_rl.core.constants import BOARD_SHAPE, FILLED_CELL_MAX, PIECE_TYPES
from brainblock_rl.core.legality import validate_board
from brainblock_rl.core.pieces import piece_index


def remaining_piece_counts(queue: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return counts for I, O, L, Z, T remaining in the queue."""

    counts = np.zeros(len(PIECE_TYPES), dtype=np.int8)
    for piece in queue:
        counts[piece_index(piece)] += 1
    return counts


def current_piece_index(queue: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return the current piece index, or ``-1`` when the queue is empty."""

    current = -1 if not queue else piece_index(queue[0])
    return np.array(current, dtype=np.int8)


def make_observation(board: np.ndarray, queue: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
    """Build a Gymnasium-compatible observation dictionary."""

    board_array = validate_board(board).astype(np.int8, copy=True)
    return {
        "board": board_array,
        "current_piece": current_piece_index(queue),
        "remaining_counts": remaining_piece_counts(queue),
    }


def make_observation_space() -> spaces.Dict:
    """Create the observation space matching ``make_observation``."""

    return spaces.Dict(
        {
            "board": spaces.Box(
                low=0,
                high=FILLED_CELL_MAX,
                shape=BOARD_SHAPE,
                dtype=np.int8,
            ),
            "current_piece": spaces.Box(
                low=-1,
                high=len(PIECE_TYPES) - 1,
                shape=(),
                dtype=np.int8,
            ),
            "remaining_counts": spaces.Box(
                low=0,
                high=2,
                shape=(len(PIECE_TYPES),),
                dtype=np.int8,
            ),
        }
    )


__all__ = [
    "current_piece_index",
    "make_observation",
    "make_observation_space",
    "remaining_piece_counts",
]
