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

from tac.agent.algorithm import policy_util
from tac.agent.net import optimisers
from tac.agent.algorithm.policy_util import epsilon_greedy
from tac.agent.net.mlp import MLPNet
from tac.utils.general import to_torch_batch, set_attr_from_dict
from tac.agent.algorithm.base import Algorithm
from tac.utils.decorator import api

matplotlib.use('TkAgg')


class Sarsa(Algorithm):

    # TODO: Remove this after implementing init_algorithm_params and init_nets
    def __init__(self, agent):

        # Overwritten in init_algorithm_params method. Included here to document the attributes
        self.gamma = None
        self.optimiser = None
        self.net = None
        self.net_name = None
        self.epsilon = None
        self.action_policy = None
        self.ready_to_train = None

        super().__init__(agent)

        # TODO: are the next 3 lines right to be commented out? If so, get rid of on_policy_reset method
        #
        # self.rewards = []
        # self.log_probs = []
        #
        # self.on_policy_reset()  # Initialise the log_probs and rewards lists

        # TODO - extra attributes - check these are ok. Implemented differently in SLM Lab (agent/algorithm/body)
        self.env = agent.env
        self.action_space = self.env.action_space

    @api
    def init_algorithm_params(self):
        """Initialise other algorithm parameters"""

        # Set default values for algorithm parameters
        set_attr_from_dict(self, {
            "action_pd_type": "default",    # TODO: check this value is supported
            "action_policy": "default",    # TODO: check this value is supported
            "explore_var_spec": None
        })

        # Overwrite default values with values from spec file
        set_attr_from_dict(self, self.algorithm_spec, keys=[
            "action_pd_type",
            "action_policy",
            "explore_var_spec",
            "gamma",
            "training_frequency",
        ])
        self.ready_to_train = False
        # The getattr here returns the matching function
        self.action_policy = getattr(policy_util, self.action_policy)
        # TODO: next two are hard-coding epsilon-greedy policy. Make this more general in future
        #  (c.f. SLM Lab implementation)
        self.epsilon = self.algorithm_spec["explore_var_spec"]["epsilon"]

    @api
    def init_nets(self, global_nets=None):
        """Initialise the neural network from the net_spec"""

        if "Recurrent" in self.net_spec.get("type"):
            raise NotImplementedError("Recurrent nets not yet supported")

        in_dim = self.agent.state_cardinality
        out_dim = self.agent.action_cardinality
        # TODO: MLP is currently hard-coded. Make this more general in future (c.f. SLM Lab implementation)
        self.net = MLPNet(self.net_spec, in_dim, out_dim)
        self.net_name = self.net_spec["type"]

        # Initialise optimiser and LR scheduler
        optimiser_class = optimisers[self.net_spec["optim_spec"]["name"]]
        self.optimiser = optimiser_class(self.net.parameters(), lr=self.net_spec["optim_spec"]["learning_rate"])

        # TODO: a couple of SLM lab lines skipped after this. Double check they are not needed.

    def on_policy_reset(self):
        """Resets the log_probs and rewards lists. Called at the start of each episode"""
        # TODO: Duplicate of SARSA method. Ultimately replace with OnPolicy Memory class, common to different algorithms
        self.log_probs = []
        self.rewards = []

    @api
    def calc_pdparam(self, state):
        """
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the
        correct outputs. The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous
        prob. dist.

        Parameters
        ----------
        state: torch.tensor
            The current state of the env

        Returns
        -------
        pdparam: torch.tensor
            The logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        """
        pdparam = self.net.forward(state)
        return pdparam

    @api
    def act(self, state):
        """
        Selects an action from the policy network's output distribution, and stores the log probability of the policy
        network's output distribution at the selected action in the log_probs list.

        Parameters
        ----------
        state: np.ndarray
            The current state of the env
        """
        # Numpy to PyTorch tensor
        state = torch.from_numpy(state.astype(np.float32))
        # Select an action from the action space using epsilon-greedy policy
        action = epsilon_greedy(state, self)
        return action.item()

    @api
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
        batch = to_torch_batch(batch, is_episodic=memory.is_episodic, device=None)

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

        # Extract the Q values for the action actually taken
        #   .long() converts the actions to long ints, which is required for indexing in the gather method of pytorch
        #   .unsqueeze(-1) adds a dimension to the end of the tensor
        #   q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)) is equivalent to q_preds[batch['actions']] in np
        #   .squeeze(-1) removes the extra dimension added by .unsqueeze(-1)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        act_next_q_preds = next_q_preds.gather(-1, batch['next_actions'].long().unsqueeze(-1)).squeeze(-1)

        # Targets, Q_tar(s, a, theta) = r + gamma * Q(s', a', theta)
        # act_q_targets = batch["rewards"] + self.gamma * next_q_preds * (1 - batch["dones"])
        act_q_targets = batch["rewards"] + self.gamma * act_next_q_preds * ~batch["dones"]

        q_loss = self.net.loss_fn(act_q_preds, act_q_targets)

        return q_loss

    @api
    def train(self, memory):
        """Completes a training step of the SARSA algorithm if it is time to train. Otherwise, does nothing"""

        if self.ready_to_train:
            batch = self.sample(memory)
            loss = self._calc_q_loss(batch)
            self.net.train_step(loss, self.optimiser)
            # reset
            self.ready_to_train = False
            return loss.item()
        else:
            return np.nan

    @api
    def update(self, state, action, reward, next_state, done):
        """
        Updates the algorithm after training.
        In SLM-Lab, this means the epsilon / tau exploration variables if there is a schedule for decaying them.
        """
        raise NotImplementedError(
            "c.f. SLM Lab implementation - currently TAC does not have a body / explore_var attribute"
        )
