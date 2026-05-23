import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.constants import BOARD_SHAPE
from brainblock_rl.core.observations import make_observation, make_observation_space


def test_observation_contains_board_current_piece_and_remaining_counts():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)
    queue = ["T", "I", "I", "O"]

    observation = make_observation(board, queue)

    assert set(observation) == {"board", "current_piece", "remaining_counts"}
    assert observation["board"].shape == BOARD_SHAPE
    assert int(observation["current_piece"]) == 4
    assert observation["remaining_counts"].tolist() == [2, 1, 0, 0, 1]


def test_observation_space_contains_observation():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)
    observation = make_observation(board, ["I", "O"])

    assert make_observation_space().contains(observation)


def test_empty_queue_uses_negative_one_current_piece():
    board = np.zeros(BOARD_SHAPE, dtype=np.int8)
    observation = make_observation(board, [])

    assert int(observation["current_piece"]) == -1
    assert observation["remaining_counts"].tolist() == [0, 0, 0, 0, 0]
