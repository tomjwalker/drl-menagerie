import numpy as np
import torch
from tac.agent.algorithm.policy_util import _random_choice, _greedy_choice, epsilon_greedy
from gymnasium.spaces import Discrete
from unittest.mock import Mock


# Mock algorithm class, for testing
class MockAlgorithm:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    @property
    def action_space(self):
        # A Gymnasium Discrete action space
        return Discrete(2)

    def calc_pdparam(self, state):
        # This method mocks the output of the policy network, acting on a torch tensor state
        # A tensor of logits
        return torch.tensor([1.0, 2.0, 3.0])


def test_random_choice():
    algorithm = MockAlgorithm(epsilon=0.1)
    action = _random_choice(algorithm)
    assert isinstance(action, torch.Tensor)
    assert action in [0, 1]


def test_greedy_choice():
    # algorithm.calc_pdparam = Mock(return_value=torch.tensor([1.0, 2.0, 3.0]))
    algorithm = MockAlgorithm(epsilon=0.1)
    state = torch.tensor([1.0])
    action = _greedy_choice(state, algorithm)
    assert isinstance(action, torch.Tensor)
    # Best action should be the index of the highest logit from the mock policy network (the third logit)
    assert action == 2


def test_epsilon_greedy():
    epsilon_values = [0.0, 1.0]    # Test with no exploration, and with full exploration
    for epsilon in epsilon_values:
        algorithm = MockAlgorithm(epsilon=epsilon)
        state = torch.tensor([1, 2, 3])
        action = epsilon_greedy(state, algorithm)
        if epsilon == 0.0:
            # If epsilon is 0, the action should be the greedy action
            assert action == _greedy_choice(state, algorithm)
        elif epsilon == 1.0:
            # If epsilon is 1, the action should be a random action
            assert action == _random_choice(algorithm)
        else:
            # If epsilon is between 0 and 1, the action should be one of the two choices
            assert action in [0, 1]
