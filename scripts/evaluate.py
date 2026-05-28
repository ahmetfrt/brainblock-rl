"""Evaluate BrainBlock baseline policies or a trained DQN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brainblock_rl.core.actions import legal_action_mask
from brainblock_rl.core.observations import make_observation
from brainblock_rl.core.rewards import VALID_REWARD_MODES
from brainblock_rl.envs.brainblock_env import BrainBlockEnv
from brainblock_rl.evaluation.metrics import aggregate_rollouts, format_metrics
from brainblock_rl.evaluation.rollout import POLICIES, Policy, RolloutResult, get_policy, run_rollouts

DQN_POLICY_NAME = "dqn"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=sorted([*POLICIES, DQN_POLICY_NAME]),
        default="legal_random",
        help="Policy to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="DQN checkpoint path. Required when --policy dqn.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for DQN evaluation: auto, cpu, cuda, etc.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Episode i uses seed + i.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=VALID_REWARD_MODES,
        default="placeholder",
        help="Reward function used by the evaluation environment.",
    )
    parser.add_argument(
        "--save-rollouts",
        type=Path,
        help="Optional CSV path for per-episode rollout traces.",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        help="Optional JSON path for aggregate evaluation metrics.",
    )
    return parser.parse_args()


def build_dqn_policy(checkpoint: Path, *, device: str) -> Policy:
    """Load a DQN checkpoint and return a deterministic rollout policy."""

    from brainblock_rl.agents.dqn_agent import DQNAgent

    agent = DQNAgent.load(checkpoint, device=device)

    def policy(env: BrainBlockEnv) -> int:
        observation = make_observation(env.board, env.queue)
        mask = legal_action_mask(env)
        return agent.select_action(observation, mask, training=False)

    return policy


def write_rollouts(path: Path, results: list[RolloutResult]) -> None:
    """Write rollout traces to CSV for later qualitative analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "total_reward",
                "episode_length",
                "success",
                "invalid_actions",
                "final_covered_area",
                "actions",
                "placed_pieces",
            ],
        )
        writer.writeheader()
        for episode_index, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "episode": episode_index,
                    "total_reward": f"{result.total_reward:.6f}",
                    "episode_length": result.episode_length,
                    "success": int(result.success),
                    "invalid_actions": result.invalid_actions,
                    "final_covered_area": result.final_covered_area,
                    "actions": json.dumps(result.actions),
                    "placed_pieces": json.dumps(result.placed_pieces),
                }
            )


def main() -> None:
    args = parse_args()
    if args.policy == DQN_POLICY_NAME:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --policy dqn")
        policy = build_dqn_policy(args.checkpoint, device=args.device)
    else:
        policy = get_policy(args.policy)

    env_factory = lambda: BrainBlockEnv(reward_mode=args.reward_mode)
    results = run_rollouts(
        env_factory,
        policy,
        episodes=args.episodes,
        seed=args.seed,
    )
    metrics = aggregate_rollouts(results)
    print(format_metrics(metrics, policy_name=args.policy))
    if args.save_metrics is not None:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        payload = metrics.to_dict()
        payload.update(
            {
                "policy": args.policy,
                "reward_mode": args.reward_mode,
                "seed": args.seed,
                "checkpoint": "" if args.checkpoint is None else str(args.checkpoint),
            }
        )
        args.save_metrics.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Metrics: {args.save_metrics}")
    if args.save_rollouts is not None:
        write_rollouts(args.save_rollouts, results)
        print(f"Rollout traces: {args.save_rollouts}")


if __name__ == "__main__":
    main()
