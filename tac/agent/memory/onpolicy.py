from tac.agent.memory.base import Memory


class OnPolicyReplay(Memory):

    def __init__(self, spec, agent):

        super().__init__(spec, agent)

        self.dones = None
        self.next_states = None
        self.rewards = None
        self.actions = None
        self.states = None
        self.training_frequency = agent.algorithm.training_frequency

        self.is_episodic = True
        # Number of experiences stored in the replay buffer
        self.size = 0
        # Number of experiences seen, cumulatively
        self.seen_size = 0
        # Declare data keys to store (initialised in reset method)
        self.data_keys = ["states", "actions", "rewards", "next_states", "dones"]

        # These next two initialised properly in reset method
        self.most_recent = None
        self.current_episode_data = None

        # TODO: hack, 24/10/23. Need to rewrite this with body/agent/algorithm/memory/net differentiation as SLMLab
        self.agent = agent

        self.reset()

    def _add_experience(self, state, action, reward, next_state, done):
        """
        Adds a single experience to the replay buffer.
        Helper method to differentiate SARSA-specific updates from general algorithm API update method.
        """
        self.most_recent = (state, action, reward, next_state, done)
        # Add the most recent experience to each of the lists which are the values of the current episode data dict
        for i, k in enumerate(self.data_keys):
            self.current_episode_data[k].append(self.most_recent[i])

        # If episode ended, add the current episode data to the replay buffer (the individual attributes matching the
        # strings in `data_keys`), then clear the current episode data dictionary
        if done:
            for k in self.data_keys:
                getattr(self, k).append(self.current_episode_data[k])
            self.current_episode_data = {k: [] for k in self.data_keys}
            # If agent has collected the desired number of experiences, it is ready to train
            if len(self.states) == self.training_frequency:
                self.agent.algorithm.ready_to_train = True

        # Track memory size and number of experiences seen
        self.size += 1
        self.seen_size += 1

    def update(self, state, action, reward, next_state, done):
        """Adds a single experience to the replay buffer."""
        self._add_experience(state, action, reward, next_state, done)

    def sample(self):
        """
        Returns all examples from memory in a single batch. Batch is stored as a dict. Keys are the names of the
        different elements of an experience, values are nested lists of the experiences. Elements are nested into
        episodes.
        e.g.
        batch = {
            "states": [[s_episode_1], [s_episode_2], ...],
            "actions": [[a_episode_1], [a_episode_2], ...],
            "rewards": [[r_episode_1], [r_episode_2], ...],
            "next_states": [[s_episode_1], [s_episode_2], ...],
            "dones": [[d_episode_1], [d_episode_2], ...],
        }

        :return: batch (dict)

        """
        batch = {k: getattr(self, k) for k in self.data_keys}
        self.reset()
        return batch

    def reset(self):
        """Resets the replay buffer to its initial state. Initialises the current episode data dictionary."""
        # Add each element of data keys as a new attribute to the replay buffer object
        for k in self.data_keys:
            setattr(self, k, [])
        # Reset / initialise the current episode data. This is a dictionary of lists, where each list is a list of
        # experiences from a single episode. The keys of the dictionary are the data keys.
        self.current_episode_data = {k: [] for k in self.data_keys}
        # Most recent experience, as a tuple
        self.most_recent = (None,) * len(self.data_keys)
        self.size = 0


class OnPolicyBatchReplay(OnPolicyReplay):

    def __init__(self, spec, agent):
        super().__init__(spec, agent)
        self.is_episodic = False

    def _add_experience(self, state, action, reward, next_state, done):
        """
        Helper method for BatchReplay. This differs from the OnPolicyReplay superclass method in that it adds
        the constituents of an experience to their respective attributes directly, rather than as nested lists (see
        .sample docstring for difference in output format).
        """
        self.most_recent = [state, action, reward, next_state, done]
        for i, k in enumerate(self.data_keys):
            getattr(self, k).append(self.most_recent[i])
        self.size += 1
        self.seen_size += 1
        # If agent has collected the desired number of experiences, it is ready to train
        if len(self.states) == self.training_frequency:
            self.agent.algorithm.ready_to_train = True

    def sample(self):
        """
        Returns all examples from memory in a single batch. Batch is stored as a dict. Keys are the names of the
        different elements of an experience, values are lists of the experiences. Elements are not nested into episodes.
        e.g.
        batch = {
            "states": [s_1, s_2, ...],
            "actions": [a_1, a_2, ...],
            "rewards": [r_1, r_2, ...],
            "next_states": [s_1, s_2, ...],
            "dones": [d_1, d_2, ...],
        }
        """
        return super().sample()
