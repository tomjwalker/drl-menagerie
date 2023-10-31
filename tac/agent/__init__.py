import torch
import numpy as np

from tac.agent import algorithm
from tac.agent.memory import memory_classes
from tac.agent.algorithm import algorithm_classes


#######################################################################################################################
# Agent API
#######################################################################################################################

class Agent:

    def __init__(self, spec, env):
        self.spec = spec
        self.name = spec["algorithm"]
        self.env = env
        self.loss = np.nan

        # Initialise algorithm
        algorithm_class = algorithm_classes[spec["algorithm"]]
        state_cardinality = self.env.observation_space.shape[0]
        action_cardinality = self.env.action_space.n
        self.algorithm = algorithm_class(spec, state_cardinality, action_cardinality)

        # Initialise memory
        memory_class = memory_classes[spec["memory"]]
        self.memory = memory_class(spec, self)

    def act(self, state):
        """Interface for the act method from the algorithm class. Given an environment state, returns an action"""
        # Gradients only calculated in algorithm.train
        with torch.no_grad():
            action = self.algorithm.act(state)
        return action

    def update(self, state, action, reward, next_state, done):

        # TODO: body.update method not replicated in TAC. Check this

        # TODO: not check for eval_lab_mode

        # Add new experience to memory
        self.memory.update(state, action, reward, next_state, done)

        # Perform a step of training (this method within the algorthm checks whether it is time to train)
        loss = self.algorithm.train(self.memory)
        # The .train() method returns np.nan as default if the time to train flag is false
        if not np.isnan(loss):
            self.loss = loss

        # TODO: explore var update not currently implemented in algorithms
        # explore_var = self.algorithm.update()

        return loss

    def save(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
