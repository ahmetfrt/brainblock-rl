"""Experience replay for DQN training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One DQN transition with the next state's legal-action mask."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_mask: np.ndarray


class ReplayBuffer:
    """Fixed-size replay buffer backed by a deque."""

    def __init__(
        self,
        capacity: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self._buffer: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self._buffer)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        """Store one transition."""

        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                next_legal_mask=np.asarray(next_legal_mask, dtype=bool),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a mini-batch as stacked NumPy arrays."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > len(self._buffer):
            raise ValueError("batch_size cannot exceed current buffer length")

        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        transitions = [self._buffer[int(index)] for index in indices]

        return {
            "states": np.stack([transition.state for transition in transitions]),
            "actions": np.array([transition.action for transition in transitions], dtype=np.int64),
            "rewards": np.array([transition.reward for transition in transitions], dtype=np.float32),
            "next_states": np.stack([transition.next_state for transition in transitions]),
            "dones": np.array([transition.done for transition in transitions], dtype=np.float32),
            "next_legal_masks": np.stack(
                [transition.next_legal_mask for transition in transitions]
            ),
        }


__all__ = ["ReplayBuffer", "Transition"]
