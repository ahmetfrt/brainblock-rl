import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainblock_rl.core.actions import decode_action, encode_action
from brainblock_rl.core.constants import ACTION_SPACE_SIZE


@pytest.mark.parametrize(
    ("orientation", "x", "y"),
    [(0, 0, 0), (0, 7, 4), (3, 2, 1), (7, 7, 4)],
)
def test_encode_decode_round_trip(orientation, x, y):
    action_id = encode_action(orientation, x, y)

    decoded = decode_action(action_id)

    assert decoded.orientation == orientation
    assert decoded.x == x
    assert decoded.y == y
    assert tuple(decoded) == (orientation, x, y)


def test_flattening_rule_matches_spec():
    assert encode_action(2, 3, 4) == 2 * 40 + 4 * 8 + 3
    assert decode_action(319) == (7, 7, 4)


@pytest.mark.parametrize("action_id", [-1, ACTION_SPACE_SIZE])
def test_decode_rejects_out_of_range_action(action_id):
    with pytest.raises(ValueError):
        decode_action(action_id)


@pytest.mark.parametrize(
    ("orientation", "x", "y"),
    [(-1, 0, 0), (8, 0, 0), (0, -1, 0), (0, 8, 0), (0, 0, -1), (0, 0, 5)],
)
def test_encode_rejects_out_of_range_components(orientation, x, y):
    with pytest.raises(ValueError):
        encode_action(orientation, x, y)
