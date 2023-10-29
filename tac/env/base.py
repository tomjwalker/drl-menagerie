from abc import ABC, abstractmethod
import time


class Clock:

    def __init__(self, max_frame=int(1e7), clock_speed=1):

        self.max_frame = max_frame
        self.clock_speed = int(clock_speed)

        # These next attributes are set in the reset method
        self.opt_step = None
        self.batch_size = None
        self.wall_t = None
        self.start_wall_t = None
        self.epi = None
        self.frame = None
        self.t = None
        self.reset()

    def reset(self):
        self.t = 0
        self.frame = 0    # i.e. total_t
        self.epi = 0
        # Register the time at which the clock was reset (in seconds)
        self.start_wall_t = time.time()
        self.wall_t = 0
        self.batch_size = 1    # Multiplier to accurately count opt steps
        self.opt_step = 0    # Count the number of optimiser updates

    def load(self, train_df):
        raise NotImplementedError

    def get(self, unit="frame"):
        raise getattr(self, unit)

    def get_elapsed_wall_time(self):
        """
        Calling time.time() again gets the current time, and subtracting the start time gives the elapsed time
        (in seconds)
        """
        return int(time.time() - self.start_wall_t)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def tick(self, unit="t"):
        if unit == "t":    # timestep
            self.t += self.clock_speed    # Reset to 0 at the start of each episode
            self.frame += self.clock_speed    # Not reset to 0 at the start of each episode, so tracks total timesteps
            self.wall_t = self.get_elapsed_wall_time()
        elif unit == "epi":    # Episode; reset timestep counter t
            self.epi += 1
            self.t = 0
        elif unit == "opt_step":    # Optimisation step
            self.opt_step += self.batch_size
        else:
            raise KeyError(f"Unit {unit} not recognised")


class BaseEnv(ABC):
    """
    Base Env class with API and helper methods.

    Use to implement env classes which are compatible with the TAC framework.
    """

    def __init__(self, spec):
        raise NotImplementedError

    def _get_spaces(self, u_env):
        raise NotImplementedError

    def _get_observable_dim(self, observation_space):
        raise NotImplementedError

    def _get_action_dim(self, action_space):
        raise NotImplementedError

    def _infer_frame_attr(self, spec):
        raise NotImplementedError

    def _infer_venv_attr(self):
        raise NotImplementedError

    def _is_discrete(self, action_space):
        raise NotImplementedError

    def _set_clock(self):
        self.clock_speed = 1 * (self.num_envs or 1)    # TODO: this line only salient for vectorised envs
        self.clock = Clock(max_frame=self.max_frame, clock_speed=self.clock_speed)

    def _set_attr_from_u_env(self, u_env):
        raise NotImplementedError

    def _update_total_reward(self, info):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the environment and returns the state"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Takes an action in the environment, and returns the next state, reward, done and info"""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Method to close and clean up the environment"""
        raise NotImplementedError

