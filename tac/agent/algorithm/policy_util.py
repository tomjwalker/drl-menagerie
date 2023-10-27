import numpy as np
import torch


def _random_choice(algorithm):
    """Selects a random action from the action space, using the Gymnasium api action_space.sample() method"""
    _action = algorithm.action_space.sample()
    # Need to cast as a torch tensor, since the policy network expects a tensor as input
    action = torch.tensor(_action)
    return action


def _greedy_choice(state, algorithm):
    """
    Output of the policy network is a set of logits over the action space. The action with the highest logit is the
    greedy action.
    """
    logits = algorithm.calc_pdparam(state)
    # TODO: Check this vs SLM lab implementation, which extends torch distributions within the policy_util module
    action = torch.argmax(logits)
    return action


def epsilon_greedy(state, algorithm):
    epsilon = algorithm.epsilon
    if epsilon > np.random.rand():
        return _random_choice(algorithm)
    else:
        return _greedy_choice(state, algorithm)
