"""DQN agent with legal-action masking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from brainblock_rl.agents.networks import DQNNetwork
from brainblock_rl.agents.replay_buffer import ReplayBuffer
from brainblock_rl.core.constants import (
    ACTION_SPACE_SIZE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    FILLED_CELL_MAX,
    PIECES_PER_TYPE,
    PIECE_TYPES,
)

CURRENT_PIECE_FEATURES = len(PIECE_TYPES) + 1
BOARD_FEATURES = BOARD_HEIGHT * BOARD_WIDTH * (FILLED_CELL_MAX + 1)
REMAINING_COUNT_FEATURES = len(PIECE_TYPES)
OBSERVATION_VECTOR_SIZE = BOARD_FEATURES + CURRENT_PIECE_FEATURES + REMAINING_COUNT_FEATURES


def flatten_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Convert the Gymnasium observation dict into a flat float32 vector.

    The board is one-hot encoded over empty plus five filled-cell values. The
    current piece uses an extra first slot for the terminal/empty-queue value
    ``-1``. Remaining piece counts are scaled to [0, 1].
    """

    board = np.asarray(observation["board"], dtype=np.int64)
    if board.shape != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(f"board observation must have shape {(BOARD_HEIGHT, BOARD_WIDTH)}")
    if board.min() < 0 or board.max() > FILLED_CELL_MAX:
        raise ValueError("board observation contains an unknown cell value")

    board_features = np.eye(FILLED_CELL_MAX + 1, dtype=np.float32)[board].reshape(-1)

    current_piece = int(np.asarray(observation["current_piece"]).item())
    if not -1 <= current_piece < len(PIECE_TYPES):
        raise ValueError(f"current_piece must be in [-1, {len(PIECE_TYPES) - 1}]")
    current_piece_features = np.zeros(CURRENT_PIECE_FEATURES, dtype=np.float32)
    current_piece_features[current_piece + 1] = 1.0

    remaining_counts = np.asarray(observation["remaining_counts"], dtype=np.float32)
    if remaining_counts.shape != (len(PIECE_TYPES),):
        raise ValueError(f"remaining_counts must have shape {(len(PIECE_TYPES),)}")
    remaining_features = remaining_counts / float(PIECES_PER_TYPE)

    return np.concatenate([board_features, current_piece_features, remaining_features]).astype(
        np.float32,
        copy=False,
    )


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for the DQN agent."""

    input_dim: int = OBSERVATION_VECTOR_SIZE
    action_dim: int = ACTION_SPACE_SIZE
    hidden_dims: tuple[int, ...] = (256, 256)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    min_replay_size: int = 1_000
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    gradient_clip_norm: float = 10.0
    device: str = "auto"


class DQNAgent:
    """Deep Q-network agent that only explores/exploits legal actions."""

    def __init__(
        self,
        config: DQNConfig | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config if config is not None else DQNConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.device = self._resolve_device(self.config.device)

        self.policy_net = DQNNetwork(
            self.config.input_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)
        self.target_net = DQNNetwork(
            self.config.input_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
        )
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, rng=self.rng)
        self.steps_done = 0
        self.updates_done = 0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def epsilon(self) -> float:
        """Return the current linearly decayed epsilon value."""

        if self.config.epsilon_decay_steps <= 0:
            return self.config.epsilon_end
        progress = min(1.0, self.steps_done / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(
        self,
        observation: dict[str, np.ndarray],
        legal_mask: np.ndarray,
        *,
        training: bool = True,
    ) -> int:
        """Select an action, restricting random and greedy choices to legal moves."""

        legal_mask = np.asarray(legal_mask, dtype=bool)
        legal_actions = np.flatnonzero(legal_mask)
        if len(legal_actions) == 0:
            return 0

        if training:
            self.steps_done += 1
            if self.rng.random() < self.epsilon():
                return int(self.rng.choice(legal_actions))

        state = torch.as_tensor(
            flatten_observation(observation),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0).detach().cpu().numpy()
        masked_q_values = np.where(legal_mask, q_values, -np.inf)
        return int(np.argmax(masked_q_values))

    def store_transition(
        self,
        observation: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        """Flatten and store a transition in replay memory."""

        self.replay_buffer.append(
            flatten_observation(observation),
            action,
            reward,
            flatten_observation(next_observation),
            done,
            next_legal_mask,
        )

    def update(self) -> dict[str, float] | None:
        """Run one DQN gradient update if enough replay data is available."""

        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch["next_states"],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_legal_masks = torch.as_tensor(
            batch["next_legal_masks"],
            dtype=torch.bool,
            device=self.device,
        )

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.masked_fill(~next_legal_masks, -1e9)
            has_legal_action = next_legal_masks.any(dim=1)
            next_max_q = torch.zeros(self.config.batch_size, dtype=torch.float32, device=self.device)
            next_max_q[has_legal_action] = next_q_values[has_legal_action].max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - dones) * next_max_q

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        self.updates_done += 1
        if self.updates_done % self.config.target_update_interval == 0:
            self.sync_target_network()

        return {
            "loss": float(loss.detach().cpu().item()),
            "mean_q": float(q_values.detach().mean().cpu().item()),
            "mean_target": float(targets.detach().mean().cpu().item()),
            "epsilon": float(self.epsilon()),
        }

    def sync_target_network(self) -> None:
        """Copy policy network weights into the target network."""

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a serializable checkpoint payload."""

        return {
            "config": asdict(self.config),
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "updates_done": self.updates_done,
        }

    def save(self, path: str | Path) -> None:
        """Save the agent checkpoint."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        rng: np.random.Generator | None = None,
        load_optimizer: bool = False,
    ) -> "DQNAgent":
        """Load an agent checkpoint saved by ``save``."""

        resolved_device = cls._resolve_device(device)
        try:
            checkpoint = torch.load(
                Path(path),
                map_location=resolved_device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(Path(path), map_location=resolved_device)
        config_payload = dict(checkpoint["config"])
        allowed_config_keys = {field.name for field in fields(DQNConfig)}
        config_payload = {
            key: value
            for key, value in config_payload.items()
            if key in allowed_config_keys
        }
        if "hidden_dims" in config_payload:
            config_payload["hidden_dims"] = tuple(config_payload["hidden_dims"])
        config_payload["device"] = device

        agent = cls(DQNConfig(**config_payload), rng=rng)
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(
            checkpoint.get("target_state_dict", checkpoint["policy_state_dict"])
        )
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps_done = int(checkpoint.get("steps_done", 0))
        agent.updates_done = int(checkpoint.get("updates_done", 0))
        agent.policy_net.eval()
        agent.target_net.eval()
        return agent


__all__ = [
    "DQNAgent",
    "DQNConfig",
    "OBSERVATION_VECTOR_SIZE",
    "flatten_observation",
]
