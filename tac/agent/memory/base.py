from abc import ABC, abstractmethod


class Memory(ABC):
    """Abstract base class to define the API methods for all memory classes."""

    def __init__(self, spec, agent):
        self.spec = spec
        self.agent = agent
        # TODO: check what "priorities" is
        self.data_keys = ["states", "actions", "rewards", "next_states", "dones", "priorities"]

    @abstractmethod
    def reset(self):
        """Method to fully reset the memory storage and related attributes."""
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Implement memory update given the full info from the latest step."""
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """Implement memory sampling mechanism."""
        raise NotImplementedError
