"""Train a masked DQN agent on the current BrainBlock reward."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brainblock_rl.agents.dqn_agent import DQNAgent, DQNConfig
from brainblock_rl.core.actions import legal_action_mask
from brainblock_rl.core.constants import TOTAL_PIECES
from brainblock_rl.core.rewards import VALID_REWARD_MODES
from brainblock_rl.envs.brainblock_env import BrainBlockEnv


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    """Parse comma-separated hidden layer sizes."""

    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dimensions cannot be empty")
    if any(dim < 1 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive")
    return dims


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short sanity-check training job with small settings.",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reward-mode",
        choices=VALID_REWARD_MODES,
        default="placeholder",
        help="Reward function to train with.",
    )
    parser.add_argument("--max-steps", type=int, default=TOTAL_PIECES + 1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "dqn_placeholder")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(256, 256))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--min-replay-size", type=int, default=1_000)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=5_000)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    return parser.parse_args()


def apply_smoke_settings(args: argparse.Namespace) -> None:
    """Shrink training settings for a quick end-to-end sanity check."""

    if not args.smoke:
        return

    args.episodes = 20
    args.output_dir = Path("runs") / "dqn_smoke"
    args.hidden_dims = (64, 64)
    args.batch_size = 8
    args.buffer_size = 1_000
    args.min_replay_size = 8
    args.target_update_interval = 25
    args.epsilon_decay_steps = 100
    args.log_interval = 1
    args.checkpoint_interval = 0


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_agent(args: argparse.Namespace, rng: np.random.Generator) -> DQNAgent:
    """Build the DQN agent from CLI arguments."""

    config = DQNConfig(
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_replay_size=args.min_replay_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        device=args.device,
    )
    return DQNAgent(config, rng=rng)


def write_args(output_dir: Path, args: argparse.Namespace) -> None:
    """Persist the exact training command settings."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.txt").open("w", encoding="utf-8") as file:
        for key, value in sorted(vars(args).items()):
            file.write(f"{key}: {value}\n")


def main() -> None:
    args = parse_args()
    apply_smoke_settings(args)
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    env = BrainBlockEnv(reward_mode=args.reward_mode)
    agent = make_agent(args, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_args(args.output_dir, args)

    log_path = args.output_dir / "training_log.csv"
    fieldnames = [
        "episode",
        "seed",
        "total_reward",
        "episode_length",
        "success",
        "invalid_actions",
        "final_covered_area",
        "epsilon",
        "loss",
    ]

    recent_returns: deque[float] = deque(maxlen=50)
    recent_successes: deque[float] = deque(maxlen=50)
    recent_covered_area: deque[float] = deque(maxlen=50)
    best_covered_area = -1
    best_success = False
    last_loss = float("nan")

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            episode_seed = args.seed + episode - 1
            observation, info = env.reset(seed=episode_seed)
            total_reward = 0.0
            episode_length = 0
            invalid_actions = 0
            terminated = False
            truncated = False

            for _ in range(args.max_steps):
                mask = legal_action_mask(env)
                action = agent.select_action(observation, mask, training=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_mask = legal_action_mask(env)
                done = terminated or truncated

                agent.store_transition(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done,
                    next_mask,
                )
                update_info = agent.update()
                if update_info is not None:
                    last_loss = update_info["loss"]

                total_reward += float(reward)
                episode_length += 1
                if info.get("legal") is False:
                    invalid_actions += 1

                observation = next_observation
                if done:
                    break
            else:
                truncated = True

            final_covered_area = int(info.get("covered_area", 0))
            success = bool(terminated and not truncated and len(env.queue) == 0 and invalid_actions == 0)
            recent_returns.append(total_reward)
            recent_successes.append(float(success))
            recent_covered_area.append(float(final_covered_area))

            writer.writerow(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "total_reward": f"{total_reward:.6f}",
                    "episode_length": episode_length,
                    "success": int(success),
                    "invalid_actions": invalid_actions,
                    "final_covered_area": final_covered_area,
                    "epsilon": f"{agent.epsilon():.6f}",
                    "loss": "" if np.isnan(last_loss) else f"{last_loss:.6f}",
                }
            )
            log_file.flush()

            improved = success and not best_success
            improved = improved or (success == best_success and final_covered_area > best_covered_area)
            if improved:
                best_success = success
                best_covered_area = final_covered_area
                agent.save(checkpoint_dir / "best.pt")

            if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                agent.save(checkpoint_dir / f"episode_{episode:06d}.pt")

            if args.log_interval > 0 and episode % args.log_interval == 0:
                mean_return = float(np.mean(recent_returns))
                mean_success = float(np.mean(recent_successes))
                mean_area = float(np.mean(recent_covered_area))
                loss_text = "nan" if np.isnan(last_loss) else f"{last_loss:.4f}"
                print(
                    "episode="
                    f"{episode} mean_return_50={mean_return:.3f} "
                    f"success_50={mean_success:.3f} covered_50={mean_area:.3f} "
                    f"epsilon={agent.epsilon():.3f} loss={loss_text}"
                )

    agent.save(checkpoint_dir / "latest.pt")
    print(f"Training log: {log_path}")
    print(f"Latest checkpoint: {checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
