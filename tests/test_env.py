import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.actions import encode_action
from brainblock_rl.core.constants import ACTION_SPACE_SIZE
from brainblock_rl.envs.brainblock_env import BrainBlockEnv
from brainblock_rl.evaluation.metrics import aggregate_rollouts
from brainblock_rl.evaluation.rollout import (
    RolloutResult,
    legal_random_policy,
    random_policy,
    rollout_episode,
)


def test_reset_returns_valid_observation_and_initial_queue():
    env = BrainBlockEnv()

    observation, info = env.reset(seed=7)

    assert env.action_space.n == ACTION_SPACE_SIZE
    assert env.observation_space.contains(observation)
    assert len(info["queue"]) == 10
    assert sum(info["remaining_counts"]) == 10
    assert np.count_nonzero(observation["board"]) == 0


def test_reset_seed_reproducibly_shuffles_queue():
    env_a = BrainBlockEnv()
    env_b = BrainBlockEnv()

    _, info_a = env_a.reset(seed=99)
    _, info_b = env_b.reset(seed=99)

    assert info_a["queue"] == info_b["queue"]


def test_legal_step_places_current_piece_and_advances_queue():
    env = BrainBlockEnv()
    env.reset(seed=1)
    first_piece = env.queue[0]

    observation, reward, terminated, truncated, info = env.step(encode_action(0, 0, 0))

    assert env.observation_space.contains(observation)
    assert reward == 4.0
    assert not terminated
    assert not truncated
    assert info["legal"] is True
    assert info["placed_piece"] == first_piece
    assert np.count_nonzero(env.board) == 4
    assert len(env.queue) == 9


def test_invalid_step_hard_terminates_without_advancing_queue():
    env = BrainBlockEnv()
    env.reset(seed=1)
    env.step(encode_action(0, 0, 0))
    queue_after_first_step = tuple(env.queue)

    _, reward, terminated, truncated, info = env.step(encode_action(0, 0, 0))

    assert reward == -1.0
    assert terminated
    assert not truncated
    assert info["legal"] is False
    assert tuple(env.queue) == queue_after_first_step


def test_episode_terminates_when_queue_becomes_empty():
    env = BrainBlockEnv()
    env.reset(seed=1)
    env.board.fill(0)
    env.queue = ["O"]

    _, reward, terminated, truncated, info = env.step(encode_action(0, 0, 0))

    assert reward == 4.0
    assert terminated
    assert not truncated
    assert info["legal"] is True
    assert env.queue == []


def test_out_of_range_action_hard_terminates():
    env = BrainBlockEnv()
    env.reset(seed=1)

    _, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE)

    assert reward == -1.0
    assert terminated
    assert not truncated
    assert info["invalid_reason"] == "action_out_of_range"


def test_legal_random_rollout_returns_structured_result():
    env = BrainBlockEnv()

    result = rollout_episode(env, legal_random_policy, seed=0)

    assert result.episode_length >= 1
    assert len(result.actions) == result.episode_length
    assert result.invalid_actions in (0, 1)
    assert len(result.placed_pieces) == result.episode_length - result.invalid_actions
    assert result.final_covered_area == 4 * len(result.placed_pieces)


def test_random_rollout_returns_structured_result():
    env = BrainBlockEnv()

    result = rollout_episode(env, random_policy, seed=0)

    assert result.episode_length >= 1
    assert len(result.actions) == result.episode_length
    assert all(0 <= action < ACTION_SPACE_SIZE for action in result.actions)
    assert result.invalid_actions >= 0
    assert result.final_covered_area == 4 * len(result.placed_pieces)


def test_metrics_aggregation():
    results = [
        RolloutResult(
            total_reward=40.0,
            episode_length=10,
            success=True,
            invalid_actions=0,
            final_covered_area=40,
            actions=list(range(10)),
            placed_pieces=["I"] * 10,
        ),
        RolloutResult(
            total_reward=7.0,
            episode_length=3,
            success=False,
            invalid_actions=1,
            final_covered_area=8,
            actions=[0, 1, 2],
            placed_pieces=["O", "T"],
        ),
    ]

    metrics = aggregate_rollouts(results)

    assert metrics.episodes == 2
    assert metrics.success_rate == 0.5
    assert metrics.mean_episodic_return == 23.5
    assert metrics.mean_episode_length == 6.5
    assert metrics.invalid_action_rate == 1 / 13
    assert metrics.mean_final_covered_area == 24.0
