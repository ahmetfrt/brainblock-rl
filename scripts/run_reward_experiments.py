"""Train and evaluate reward-mode DQN experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rewards",
        nargs="+",
        choices=("sparse", "shaped", "terminal", "shaped_v2"),
        default=("terminal", "shaped_v2"),
        help="Reward modes to train and evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(0, 1, 2, 3, 4),
        help="Training seeds for each reward mode.",
    )
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs") / "reward_experiments",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device forwarded to training/evaluation.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training/evaluation steps whose output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    """Run a subprocess command from the repository root."""

    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def train_command(
    *,
    reward_mode: str,
    seed: int,
    episodes: int,
    output_dir: Path,
    device: str,
) -> list[str]:
    """Build the train command for one experiment."""

    return [
        sys.executable,
        "scripts/train.py",
        "--reward-mode",
        reward_mode,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]


def evaluate_command(
    *,
    reward_mode: str,
    checkpoint: Path,
    episodes: int,
    seed: int,
    metrics_path: Path,
    rollouts_path: Path,
    device: str,
) -> list[str]:
    """Build the evaluation command for one trained checkpoint."""

    return [
        sys.executable,
        "scripts/evaluate.py",
        "--policy",
        "dqn",
        "--checkpoint",
        str(checkpoint),
        "--reward-mode",
        reward_mode,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--device",
        device,
        "--save-metrics",
        str(metrics_path),
        "--save-rollouts",
        str(rollouts_path),
    ]


def write_summary(output_root: Path, metrics_paths: list[Path]) -> Path:
    """Combine per-run metrics JSON files into one CSV summary."""

    summary_path = output_root / "comparison_summary.csv"
    rows = []
    for metrics_path in metrics_paths:
        if metrics_path.exists():
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
            match = re.search(r"seed_(\d+)", str(metrics_path))
            row["train_seed"] = "" if match is None else int(match.group(1))
            row["eval_seed"] = row.pop("seed", "")
            rows.append(row)

    fieldnames = [
        "reward_mode",
        "train_seed",
        "eval_seed",
        "episodes",
        "success_rate",
        "mean_episodic_return",
        "std_episodic_return",
        "mean_episode_length",
        "invalid_action_rate",
        "mean_final_covered_area",
        "checkpoint",
    ]
    rows.sort(key=lambda row: (row.get("reward_mode", ""), row.get("train_seed", "")))
    output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return summary_path


def main() -> None:
    args = parse_args()
    metrics_paths: list[Path] = []

    for reward_mode in args.rewards:
        for seed in args.seeds:
            run_dir = args.output_root / reward_mode / f"seed_{seed}"
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            metrics_path = run_dir / "evaluation_metrics.json"
            rollouts_path = run_dir / "evaluation_rollouts.csv"

            if not (args.skip_existing and checkpoint.exists()):
                run_command(
                    train_command(
                        reward_mode=reward_mode,
                        seed=seed,
                        episodes=args.train_episodes,
                        output_dir=run_dir,
                        device=args.device,
                    ),
                    dry_run=args.dry_run,
                )

            if not (args.skip_existing and metrics_path.exists()):
                run_command(
                    evaluate_command(
                        reward_mode=reward_mode,
                        checkpoint=checkpoint,
                        episodes=args.eval_episodes,
                        seed=args.eval_seed,
                        metrics_path=metrics_path,
                        rollouts_path=rollouts_path,
                        device=args.device,
                    ),
                    dry_run=args.dry_run,
                )
            metrics_paths.append(metrics_path)

    if not args.dry_run:
        summary_path = write_summary(args.output_root, metrics_paths)
        print(f"Comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
