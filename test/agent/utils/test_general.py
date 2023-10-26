import pytest
import torch

from tac.utils.general import to_torch_batch


def test_to_torch_batch_is_episodic_true():
    is_episodic = True

    batch = {
        "states": [[0, 1, 5], [20]],
        "actions": [[0, 2, 6], [21]],
        "rewards": [[0, 3, 7], [22]],
        "next_states": [[0, 4, 8], [23]],
        "dones": [[False, False, True], [True]],
    }

    desired_output = {
        "states": torch.tensor([0, 1, 5, 20], dtype=torch.int32),
        "actions": torch.tensor([0, 2, 6, 21], dtype=torch.int32),
        "rewards": torch.tensor([0, 3, 7, 22], dtype=torch.int32),
        "next_states": torch.tensor([0, 4, 8, 23], dtype=torch.int32),
        "dones": torch.tensor([False, False, True, True]),
    }

    output = to_torch_batch(batch, is_episodic)

    for k, v in output.items():
        assert torch.testing.assert_allclose(v, desired_output[k]) is None


def test_to_torch_batch_is_episodic_false():
    is_episodic = False

    batch = {
        "states": [0, 1, 5, 20],
        "actions": [0, 2, 6, 21],
        "rewards": [0, 3, 7, 22],
        "next_states": [0, 4, 8, 23],
        "dones": [False, False, True, True],
    }

    desired_output = {
        "states": torch.tensor([0, 1, 5, 20], dtype=torch.int32),
        "actions": torch.tensor([0, 2, 6, 21], dtype=torch.int32),
        "rewards": torch.tensor([0, 3, 7, 22], dtype=torch.int32),
        "next_states": torch.tensor([0, 4, 8, 23], dtype=torch.int32),
        "dones": torch.tensor([False, False, True, True]),
    }

    output = to_torch_batch(batch, is_episodic)

    for k, v in output.items():
        assert torch.testing.assert_allclose(v, desired_output[k]) is None

