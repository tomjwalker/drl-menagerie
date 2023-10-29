import pytest
import torch

from tac.utils.general import to_torch_batch, set_attr_from_dict


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


def test_set_attr_from_dict_no_keys():
    class TestClass:
        pass

    obj = TestClass()
    attr_dict = {"a": 1, "b": 2, "c": 3}

    output = set_attr_from_dict(obj, attr_dict)

    assert output.a == 1
    assert output.b == 2
    assert output.c == 3


def test_set_attr_from_dict_with_keys():
    class TestClass:
        pass

    obj = TestClass()
    attr_dict = {"a": 1, "b": 2, "c": 3}

    output = set_attr_from_dict(obj, attr_dict, keys=["a", "b"])

    assert output.a == 1
    assert output.b == 2
    assert not hasattr(output, "c")
