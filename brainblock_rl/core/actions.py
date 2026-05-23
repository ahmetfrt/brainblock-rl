"""Action encoding and decoding for the fixed BrainBlock action space."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from brainblock_rl.core.constants import (
    ACTION_SPACE_SIZE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ORIENTATION_COUNT,
)


class Action(NamedTuple):
    """Decoded action components."""

    orientation: int
    x: int
    y: int


def encode_action(orientation: int, x: int, y: int) -> int:
    """Flatten ``orientation, x, y`` into a discrete action id."""

    if not 0 <= orientation < ORIENTATION_COUNT:
        raise ValueError(f"Orientation must be in [0, 7], got {orientation}")
    if not 0 <= x < BOARD_WIDTH:
        raise ValueError(f"x must be in [0, {BOARD_WIDTH - 1}], got {x}")
    if not 0 <= y < BOARD_HEIGHT:
        raise ValueError(f"y must be in [0, {BOARD_HEIGHT - 1}], got {y}")

    return orientation * (BOARD_WIDTH * BOARD_HEIGHT) + y * BOARD_WIDTH + x


def decode_action(action_id: int) -> Action:
    """Unflatten an action id into ``orientation, x, y`` components."""

    action = int(action_id)
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ValueError(f"Action id must be in [0, {ACTION_SPACE_SIZE - 1}], got {action_id}")

    orientation = action // (BOARD_WIDTH * BOARD_HEIGHT)
    rem = action % (BOARD_WIDTH * BOARD_HEIGHT)
    y = rem // BOARD_WIDTH
    x = rem % BOARD_WIDTH
    return Action(orientation=orientation, x=x, y=y)


def legal_action_mask(env: object) -> np.ndarray:
    """Return a length-320 boolean mask for legal actions in an environment.

    The mask is an auxiliary helper only: the environment action space remains
    ``Discrete(320)``. If the current episode is done or no queue-head piece is
    available, every entry is ``False``.
    """

    from brainblock_rl.core.legality import is_legal_placement

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    if getattr(env, "_terminated", False):
        return mask

    queue = getattr(env, "queue", None)
    if not queue:
        return mask

    board = getattr(env, "board")
    piece = queue[0]
    for orientation in range(ORIENTATION_COUNT):
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if is_legal_placement(board, piece, orientation, x, y):
                    mask[encode_action(orientation, x, y)] = True
    return mask


flatten_action = encode_action
unflatten_action = decode_action

__all__ = [
    "Action",
    "decode_action",
    "encode_action",
    "flatten_action",
    "legal_action_mask",
    "unflatten_action",
]
