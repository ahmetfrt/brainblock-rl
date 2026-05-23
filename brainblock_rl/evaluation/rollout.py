"""Reusable baseline policies and rollout helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from brainblock_rl.core.actions import legal_action_mask
from brainblock_rl.core.rewards import covered_area

Policy = Callable[[gym.Env], int]


@dataclass(frozen=True)
class RolloutResult:
    """Structured result for one completed episode."""

    total_reward: float
    episode_length: int
    success: bool
    invalid_actions: int
    final_covered_area: int
    actions: list[int] = field(default_factory=list)
    placed_pieces: list[str] = field(default_factory=list)


def random_policy(env: gym.Env) -> int:
    """Sample uniformly from the environment action space."""

    return int(env.action_space.sample())


def legal_random_policy(env: gym.Env) -> int:
    """Sample uniformly from currently legal actions.

    If no legal action exists, this falls back to the full action space so the
    environment can terminate through its invalid-action rule.
    """

    mask = legal_action_mask(env)
    legal_actions = np.flatnonzero(mask)
    if len(legal_actions) == 0:
        return random_policy(env)

    rng = getattr(env, "np_random", np.random.default_rng())
    return int(rng.choice(legal_actions))


POLICIES: dict[str, Policy] = {
    "random": random_policy,
    "legal_random": legal_random_policy,
}


def get_policy(name: str) -> Policy:
    """Return a baseline policy by name."""

    try:
        return POLICIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICIES))
        raise ValueError(f"Unknown policy {name!r}. Choices: {choices}") from exc


def rollout_episode(
    env: gym.Env,
    policy: Policy,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> RolloutResult:
    """Run one episode and return a structured rollout result."""

    env.reset(seed=seed)

    total_reward = 0.0
    episode_length = 0
    invalid_actions = 0
    actions: list[int] = []
    placed_pieces: list[str] = []
    terminated = False
    truncated = False
    info: dict[str, object] = {"covered_area": covered_area(getattr(env, "board"))}

    while not (terminated or truncated):
        if max_steps is not None and episode_length >= max_steps:
            truncated = True
            break

        action = int(policy(env))
        _, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        episode_length += 1
        actions.append(action)

        if info.get("legal") is False:
            invalid_actions += 1
        placed_piece = info.get("placed_piece")
        if placed_piece is not None:
            placed_pieces.append(str(placed_piece))

    queue = getattr(env, "queue", ())
    success = bool(terminated and not truncated and len(queue) == 0 and invalid_actions == 0)
    final_covered_area = int(info.get("covered_area", covered_area(getattr(env, "board"))))

    return RolloutResult(
        total_reward=total_reward,
        episode_length=episode_length,
        success=success,
        invalid_actions=invalid_actions,
        final_covered_area=final_covered_area,
        actions=actions,
        placed_pieces=placed_pieces,
    )


def run_rollouts(
    env_factory: Callable[[], gym.Env],
    policy: Policy,
    *,
    episodes: int,
    seed: int = 0,
) -> list[RolloutResult]:
    """Run multiple episodes with deterministic per-episode seeds."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    results: list[RolloutResult] = []
    for episode_index in range(episodes):
        env = env_factory()
        results.append(rollout_episode(env, policy, seed=seed + episode_index))
    return results


__all__ = [
    "POLICIES",
    "Policy",
    "RolloutResult",
    "get_policy",
    "legal_random_policy",
    "random_policy",
    "rollout_episode",
    "run_rollouts",
]
