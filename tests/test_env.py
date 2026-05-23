import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.actions import encode_action
from brainblock_rl.core.constants import ACTION_SPACE_SIZE
from brainblock_rl.envs.brainblock_env import BrainBlockEnv


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
