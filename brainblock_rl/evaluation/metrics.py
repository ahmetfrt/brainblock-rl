"""Evaluation metric aggregation for BrainBlock rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence

import numpy as np

from brainblock_rl.evaluation.rollout import RolloutResult


@dataclass(frozen=True)
class EvaluationMetrics:
    """Summary metrics computed over a set of episode rollouts."""

    episodes: int
    success_rate: float
    mean_episodic_return: float
    std_episodic_return: float
    mean_episode_length: float
    invalid_action_rate: float
    mean_final_covered_area: float

    def to_dict(self) -> dict[str, float | int]:
        """Return metrics as a plain dictionary."""

        return asdict(self)


def aggregate_rollouts(results: Sequence[RolloutResult]) -> EvaluationMetrics:
    """Aggregate rollout results into standard evaluation metrics."""

    if not results:
        raise ValueError("At least one rollout result is required")

    returns = np.array([result.total_reward for result in results], dtype=np.float64)
    lengths = np.array([result.episode_length for result in results], dtype=np.float64)
    successes = np.array([result.success for result in results], dtype=np.float64)
    invalid_actions = np.array([result.invalid_actions for result in results], dtype=np.float64)
    covered_areas = np.array([result.final_covered_area for result in results], dtype=np.float64)

    total_steps = float(lengths.sum())
    invalid_action_rate = 0.0 if total_steps == 0.0 else float(invalid_actions.sum() / total_steps)

    return EvaluationMetrics(
        episodes=len(results),
        success_rate=float(successes.mean()),
        mean_episodic_return=float(returns.mean()),
        std_episodic_return=float(returns.std()),
        mean_episode_length=float(lengths.mean()),
        invalid_action_rate=invalid_action_rate,
        mean_final_covered_area=float(covered_areas.mean()),
    )


def format_metrics(metrics: EvaluationMetrics, *, policy_name: str | None = None) -> str:
    """Format metrics as a readable CLI summary."""

    lines = ["Evaluation summary"]
    if policy_name is not None:
        lines.append(f"Policy: {policy_name}")
    lines.extend(
        [
            f"Episodes: {metrics.episodes}",
            f"Success rate: {metrics.success_rate:.3f}",
            f"Mean episodic return: {metrics.mean_episodic_return:.3f}",
            f"Std episodic return: {metrics.std_episodic_return:.3f}",
            f"Mean episode length: {metrics.mean_episode_length:.3f}",
            f"Invalid-action rate: {metrics.invalid_action_rate:.3f}",
            f"Mean final covered area: {metrics.mean_final_covered_area:.3f}",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "EvaluationMetrics",
    "aggregate_rollouts",
    "format_metrics",
]
