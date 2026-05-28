"""Gymnasium environment for BrainBlock RL."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from brainblock_rl.core.actions import decode_action, legal_action_mask
from brainblock_rl.core.constants import (
    ACTION_SPACE_SIZE,
    BOARD_SHAPE,
    CELLS_PER_PIECE,
    EMPTY_CELL,
    INDEX_TO_PIECE,
)
from brainblock_rl.core.legality import is_legal_placement, place_piece
from brainblock_rl.core.observations import (
    make_observation,
    make_observation_space,
    remaining_piece_counts,
)
from brainblock_rl.core.pieces import create_initial_queue
from brainblock_rl.core.rewards import compute_step_reward, covered_area, validate_reward_mode


class BrainBlockEnv(gym.Env):
    """BrainBlock tetromino placement environment.

    The board is stored consistently as ``board[y, x]``. Each step attempts to
    place only the current queue-head piece using a flattened action id.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        reward_mode: str = "placeholder",
    ) -> None:
        if render_mode not in (None, "ansi", "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.reward_mode = validate_reward_mode(reward_mode)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = make_observation_space()

        self.board = np.full(BOARD_SHAPE, EMPTY_CELL, dtype=np.int8)
        self.queue: list[str] = []
        self._terminated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the board and create a shuffled 10-piece queue."""

        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

        self.board = np.full(BOARD_SHAPE, EMPTY_CELL, dtype=np.int8)
        self.queue = create_initial_queue(self.np_random)
        self._terminated = False

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply an action using hard termination for invalid placements."""

        if self._terminated:
            info = self._get_info()
            info["episode_already_done"] = True
            return self._get_obs(), 0.0, True, False, info

        if not self.action_space.contains(action):
            self._terminated = True
            info = self._get_info()
            info.update({"legal": False, "invalid_reason": "action_out_of_range"})
            return (
                self._get_obs(),
                compute_step_reward(False, reward_mode=self.reward_mode),
                True,
                False,
                info,
            )

        decoded = decode_action(int(action))
        if not self.queue:
            self._terminated = True
            info = self._get_info()
            info.update({"legal": False, "invalid_reason": "empty_queue", "action": decoded})
            return (
                self._get_obs(),
                compute_step_reward(False, reward_mode=self.reward_mode),
                True,
                False,
                info,
            )

        current_piece = self.queue[0]
        if not is_legal_placement(
            self.board,
            current_piece,
            decoded.orientation,
            decoded.x,
            decoded.y,
        ):
            self._terminated = True
            info = self._get_info()
            info.update(
                {
                    "action": decoded,
                    "current_piece": current_piece,
                    "invalid_reason": "illegal_placement",
                    "legal": False,
                }
            )
            return (
                self._get_obs(),
                compute_step_reward(False, reward_mode=self.reward_mode),
                True,
                False,
                info,
            )

        place_piece(
            self.board,
            current_piece,
            decoded.orientation,
            decoded.x,
            decoded.y,
            copy=False,
        )
        self.queue.pop(0)
        self._terminated = len(self.queue) == 0
        next_legal_action_count = (
            None if self._terminated else int(np.count_nonzero(legal_action_mask(self)))
        )

        reward = compute_step_reward(
            True,
            CELLS_PER_PIECE,
            terminated=self._terminated,
            reward_mode=self.reward_mode,
            board=self.board,
            next_legal_action_count=next_legal_action_count,
        )
        info = self._get_info()
        info.update(
            {
                "action": decoded,
                "legal": True,
                "next_legal_action_count": next_legal_action_count,
                "placed_cells": CELLS_PER_PIECE,
                "placed_piece": current_piece,
            }
        )
        return self._get_obs(), reward, self._terminated, False, info

    def render(self) -> str | None:
        """Render the board as text."""

        board_text = self._board_to_string()
        if self.render_mode == "human":
            print(board_text)
            return None
        return board_text

    def _get_obs(self) -> dict[str, np.ndarray]:
        return make_observation(self.board, self.queue)

    def _get_info(self) -> dict[str, Any]:
        return {
            "covered_area": covered_area(self.board),
            "queue": tuple(self.queue),
            "remaining_counts": remaining_piece_counts(self.queue),
            "reward_mode": self.reward_mode,
        }

    def _board_to_string(self) -> str:
        symbols = {EMPTY_CELL: "."}
        symbols.update({index + 1: piece for index, piece in INDEX_TO_PIECE.items()})
        return "\n".join(
            " ".join(symbols[int(value)] for value in row)
            for row in self.board
        )
