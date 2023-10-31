"""
SARSA initial implementation.

TODO:
    - Refactor to fit standard algorithm template
    - Add support for continuous action spaces
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import gymnasium as gym

from tac.agent.net import optimisers
from tac.agent.algorithm.policy_util import epsilon_greedy
from tac.agent.net.mlp import MLPNet
from tac.utils.general import to_torch_batch

matplotlib.use('TkAgg')


class Sarsa:

    def __init__(self, spec_dict, input_size, output_size):
        self.rewards = []
        self.log_probs = []

        self.net = MLPNet(spec_dict, input_size, output_size)
        self.on_policy_reset()  # Initialise the log_probs and rewards lists

        self.optimiser = optimisers.get(spec_dict["optimiser"])(self.net.parameters(), lr=spec_dict["learning_rate"])
        self.discount_factor = spec_dict["gamma"]
        self.epsilon = spec_dict["epsilon"]

        # TODO - extra attributes - check these are ok. Implemented differently in SLM Lab (agent/algorithm/body)
        self.environment = gym.make(spec_dict["environment"])
        self.action_space = self.environment.action_space
        self.ready_to_train = False

        self.training_frequency = spec_dict["training_frequency"]

    def on_policy_reset(self):
        """Resets the log_probs and rewards lists. Called at the start of each episode"""
        # TODO: Duplicate of SARSA method. Ultimately replace with OnPolicy Memory class, common to different algorithms
        self.log_probs = []
        self.rewards = []

    def calc_pdparam(self, state):
        """
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the
        correct outputs. The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous
        prob. dist.
        """
        pdparam = self.net.forward(state)
        return pdparam

    def act(self, state):
        """
        Selects an action from the policy network's output distribution, and stores the log probability of the policy
        network's output distribution at the selected action in the log_probs list.
        """
        # Numpy to PyTorch tensor
        state = torch.from_numpy(state.astype(np.float32))
        # Select an action from the action space using epsilon-greedy policy
        action = epsilon_greedy(state, self)
        return action.item()

    @staticmethod
    def sample(memory):
        """Sample a batch from memory"""

        # Some way of getting the batch from memory. batch = memory.sample()
        batch = memory.sample()

        # Initialise a new element to the batch dictionary, for the next actions
        batch["next_actions"] = np.zeros_like(batch["actions"])

        # Fill in historical actions; batch["next_actions"][:-1] = batch["actions"][1:]
        batch["next_actions"][:-1] = batch["actions"][1:]

        # Look at util.to_torch_batch and replicate the method here
        # TODO: check whether is_episodic should be True or False
        batch = to_torch_batch(batch, is_episodic=True, device=None)

        return batch

    def _calc_q_loss(self, batch):
        """Calculate the loss for the Q-network"""
        states = batch["states"]
        next_states = batch["next_states"]
        # Get Q(s, a, theta)
        q_preds = self.net.forward(states)
        # "with torch.no_grad()" means that the operations within the block are not tracked by the autograd engine.
        with torch.no_grad():
            # Get Q(s', a', theta)
            next_q_preds = self.net.forward(next_states)
        # TODO: skipping .gather methods in SLM Sarsa version temporarily until know what they do (lines 123-124)
        # Targets, Q_tar(s, a, theta) = r + gamma * Q(s', a', theta)
        act_q_targets = batch["rewards"] + self.discount_factor * next_q_preds * (1 - batch["dones"])

        # Calculate MSE loss. L(theta) = (1/N) * sum_over_batch((Q_tar(s, a, theta) - Q(s, a, theta))^2)
        # Torch provides a function for this
        # TODO: This is hard-coding a specific loss function. Make this more general, with a setting in the spec file
        mse_criterion = nn.MSELoss()
        # TODO: requires objects to be flattened?
        q_loss = mse_criterion(q_preds, act_q_targets)
        return q_loss

    def train(self, memory):
        """Completes a training step of the SARSA algorithm if it is time to train. Otherwise, does nothing"""

        # TODO: work out what clock is

        if self.ready_to_train:
            batch = self.sample(memory)
            loss = self._calc_q_loss(batch)
            self.net.train_step(loss, self.optimiser)
            # reset
            self.ready_to_train = False
            return loss.item()
        else:
            return np.nan

    def update(self, state, action, reward, next_state, done):
        """Updates the agent after training"""
        raise NotImplementedError(
            "c.f. SLM Lab implementation - currently TAC does not have a body / explore_var attribute"
        )


# DEBUG

from tac.spec.sarsa.temp_spec import spec
from tac.agent.memory.onpolicy import OnPolicyBatchReplay

if __name__ == "__main__":

    env = gym.make(spec["environment"])
    state_cardinality = env.observation_space.shape[0]
    action_cardinality = env.action_space.n

    sarsa = Sarsa(spec_dict=spec, input_size=1, output_size=1)

    memory = OnPolicyBatchReplay(agent=sarsa)

    # batch = memory.sample()
    batch = {
        "states": [[0.1, 0.2, 0.3], [-0.1, -0.2]],
        "actions": [[0, 0, 1], [1, 0]],
        "rewards": [[1, 1, 1], [1, 1]],
        "next_states": [[0.2, 0.3, 0.2], [-0.2, -0.1]],
        "dones": [[False, False, True], [False, True]],
    }
    batch = to_torch_batch(batch, is_episodic=True, device=None)

    sarsa._calc_q_loss(batch)
