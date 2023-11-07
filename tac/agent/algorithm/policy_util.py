import numpy as np
import torch


def _random_choice(algorithm):
    """Selects a random action from the action space, using the Gymnasium api action_space.sample() method"""

    # The action space is a gym.spaces object, which has a sample() method, which returns a sample action (an element of
    # the action space)
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
    """
    Selects an action from the action space using epsilon-greedy policy. If epsilon > random number, then select a
    random action. Otherwise, select the greedy action.


    Parameters
    ----------
    state: torch.tensor
        The current state of the env
    algorithm: tac.agent.algorithm.base.Algorithm
        The algorithm object, which contains the policy network. It can be an instance of any Algorithm subclass.

    Returns
    -------
    action: torch.tensor
    """
    epsilon = algorithm.epsilon
    random_float = np.random.rand()
    if random_float < epsilon:
        return _random_choice(algorithm)
    else:
        return _greedy_choice(state, algorithm)
