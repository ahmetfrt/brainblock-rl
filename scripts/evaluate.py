"""Evaluate BrainBlock baseline policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brainblock_rl.envs.brainblock_env import BrainBlockEnv
from brainblock_rl.evaluation.metrics import aggregate_rollouts, format_metrics
from brainblock_rl.evaluation.rollout import POLICIES, get_policy, run_rollouts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=sorted(POLICIES),
        default="legal_random",
        help="Baseline policy to evaluate.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = get_policy(args.policy)
    results = run_rollouts(
        BrainBlockEnv,
        policy,
        episodes=args.episodes,
        seed=args.seed,
    )
    metrics = aggregate_rollouts(results)
    print(format_metrics(metrics, policy_name=args.policy))


if __name__ == "__main__":
    main()
