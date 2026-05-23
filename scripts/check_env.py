"""Run a few smoke-test episodes for the BrainBlock environment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brainblock_rl.core.actions import encode_action
from brainblock_rl.core.constants import BOARD_HEIGHT, BOARD_WIDTH, ORIENTATION_COUNT
from brainblock_rl.core.legality import is_legal_placement
from brainblock_rl.envs.brainblock_env import BrainBlockEnv


def legal_actions(env: BrainBlockEnv) -> list[int]:
    """Enumerate currently legal action ids for the queue-head piece."""

    if not env.queue:
        return []

    piece = env.queue[0]
    actions: list[int] = []
    for orientation in range(ORIENTATION_COUNT):
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if is_legal_placement(env.board, piece, orientation, x, y):
                    actions.append(encode_action(orientation, x, y))
    return actions


def run_episode(env: BrainBlockEnv, episode_index: int) -> None:
    """Run one episode with random legal placements until done."""

    observation, info = env.reset(seed=episode_index)
    terminated = False
    truncated = False
    total_reward = 0.0
    step_index = 0

    print(f"Episode {episode_index}")
    print(env.render())
    print(f"Initial queue: {info['queue']}")
    print(f"Initial observation keys: {tuple(observation)}")

    while not (terminated or truncated):
        choices = legal_actions(env)
        if choices:
            action = int(env.np_random.choice(choices))
        else:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_index += 1

        print(
            f"Step {step_index}: action={action}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}, legal={info.get('legal')}"
        )
        print(env.render())

    print(f"Final covered area: {info['covered_area']}")
    print(f"Total reward: {total_reward}")
    print()


def main() -> None:
    env = BrainBlockEnv(render_mode="ansi")
    for episode_index in range(3):
        run_episode(env, episode_index)


if __name__ == "__main__":
    main()
