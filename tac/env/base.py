from abc import ABC, abstractmethod
import time

import numpy as np

from gymnasium import spaces

from tac.utils.general import set_attr_from_dict
from tac.utils.general import get_class_name


def set_gym_space_attr(gym_space):
    """
    Set missing gym space attributes, for standardisation across different gym space types.

    Spaces in Gymnasium describe mathematical sets, and are used to specify valid actions (`self.action_space`) and
    observations (`self.observation_space`).

    e.g. Three possible actions, (0, 1, 2) and observations are vectors in the 2D unit cube:
    -> env.action_space = spaces.Discrete(3)
    -> env.observation_space = spaces.Box(low=0, high=1, shape=(2,))

    Types of gym spaces:
    - Box: A (possibly unbounded) box in R^n. Specifically, a Box represents the Cartesian product of n closed
        intervals. Each interval has the form of one of [a, b], (-oo, b], [a, oo), or (-oo, oo).
    - Discrete: A discrete space in {0, 1, ..., n-1}. E.g. for n=3, valid actions are (0, 1, 2).
    - MultiBinary: An n-shape binary space. Elements of this space are binary arrays of a shape which is specified
        during construction. E.g. if obs_space = MultiBinary([3, 2]), then an observation might be
        array([[0, 1], [1, 0], [0, 0]]).
    - MultiDiscrete: Represents the cartesian product of arbitrary Discrete spaces. Useful for game controllers or
        keyboards where each key can be represented as a discrete action space. E.g. Nintendo Game Controller:
            1. Arrow Keys: Discrete(5): NOOP(0), UP(1), RIGHT(2), DOWN(3), LEFT(4)
            2. Button A: Discrete(2): NOOP(0), PRESSED(1)
            3. Button B: Discrete(2): NOOP(0), PRESSED(1)
        - This would be represented as MultiDiscrete([5, 2, 2]).
    """
    if isinstance(gym_space, spaces.Box):
        setattr(gym_space, "is_discrete", False)
    elif isinstance(gym_space, spaces.Discrete):
        setattr(gym_space, "is_discrete", True)
        # TODO: Different from SLM-Lab - accommodates start parameter. Check these attributes exist in gym_space
        setattr(gym_space, "low", gym_space.start)
        setattr(gym_space, "high", (gym_space.start + gym_space.n))
    elif isinstance(gym_space, spaces.MultiBinary):
        setattr(gym_space, "is_discrete", True)
        setattr(gym_space, "low", np.full(gym_space.shape, 0))
        setattr(gym_space, "high", np.full(gym_space.shape, 2))
    elif isinstance(gym_space, spaces.MultiDiscrete):
        setattr(gym_space, "is_discrete", True)
        setattr(gym_space, "low", np.zeros_like(gym_space.nvec))
        setattr(gym_space, "high", np.array(gym_space.nvec))
    else:
        raise TypeError(f"Unrecognised gym space type: {type(gym_space)}")


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
        return getattr(self, unit)

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

        self.env_spec = spec["environment_spec"]
        self.meta_spec = spec["meta_spec"]

        # Set default attributes
        set_attr_from_dict(self, dict(
            eval_frequency=10000,
            log_frequency=10000,
            frame_op=None,
            frame_op_len=None,
            normalise_state=False,
            reward_scale=None,
            num_envs=1,
        ))

        # Set attributes from specs
        self.name = None    # Required. Set with _set_attr_from_dict method below
        self.max_frame = None    # Required. Set with _set_attr_from_dict method below
        # self.env = None    # Required. Set with _set_attr_from_dict method below
        self.max_t = None    # Optional. Stated explicitly for linting purposes (in an OR clause in OpenAIEnv)
        set_attr_from_dict(self, self.meta_spec, [
            "eval_frequency",
            "log_frequency",
        ])
        set_attr_from_dict(self, self.env_spec, [
            "name",
            "frame_op",
            "frame_op_len",
            "normalise_state",
            "reward_scale",
            "num_envs",
            "max_t",
            "max_frame",
            # "env",    # TODO: replace with env_spec.name
        ])

        self._set_clock()
        self.done = False
        self.total_reward = np.nan

        # TODO: These next attributes will only be available once the helper methods have been implemented
        # if utils.general.get_lab_mode() == "eval":
        #     self.num_envs = ps.get(spec, "meta.rigorous_eval")
        # self.to_render = utils.general.to_render()
        # self._infer_frame_attr(spec)
        # self._infer_venv_attr()

    @staticmethod
    def _get_spaces(u_env):
        """Helper to set the extra attributes to the observation and action spaces, then return them"""
        observation_space = u_env.observation_space
        action_space = u_env.action_space
        set_gym_space_attr(observation_space)
        set_gym_space_attr(action_space)
        return observation_space, action_space

    @staticmethod
    def _get_observable_dim(observation_space):
        """Get the observable dimension for an agent in the env"""
        state_dim = observation_space.shape
        if isinstance(observation_space, spaces.MultiDiscrete):
            # `.nvec` is an attribute which returns a NumPy array containing the number of discrete values for each
            # dimension of the space.
            state_dim = observation_space.nvec.tolist()
        if len(state_dim) == 1:
            # If the state is 1D, convert to a scalar
            state_dim = state_dim[0]
        return {"state_dim": state_dim}

    @staticmethod
    def _get_action_dim(action_space):
        """Get the action dim for an action_space for agent to use"""
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, "Only 1D action spaces are supported"
            action_dim = action_space.shape[0]
        elif isinstance(action_space, (spaces.Discrete, spaces.MultiBinary)):
            action_dim = action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            action_dim = action_space.nvec.tolist()
        else:
            raise TypeError(f"Unrecognised action space type: {type(action_space)}")
        return {"action_dim": action_dim}

    def _infer_frame_attr(self, spec):
        raise NotImplementedError

    def _infer_venv_attr(self):
        raise NotImplementedError

    @staticmethod
    def is_discrete(action_space):
        """Check whether the action space is discrete. All except Box are discrete"""
        return get_class_name(action_space) != "Box"

    def _set_clock(self):
        # If vectorised environments, tick with a multiple of num_envs to properly count frames
        self.clock_speed = 1 * (self.num_envs or 1)    # TODO: this line only salient for vectorised envs
        self.clock = Clock(max_frame=self.meta_spec["max_frame"], clock_speed=self.clock_speed)

    def _set_attr_from_u_env(self, u_env):
        """Set the observation and action dimensions, and the action type, from u_env"""
        self.observation_space, self.action_space = self._get_spaces(u_env)
        self.observable_dim = self._get_observable_dim(self.observation_space)
        self.action_dim = self._get_action_dim(self.action_space)
        self.is_discrete(self.action_space)

    def _update_total_reward(self, info):
        """Extract total_reward from info (set in wrapper) into self.total_reward"""
        if isinstance(info, dict):
            # `info` of dict type implies single env
            self.total_reward = info["total_reward"]
        else:
            # For vectorised environments, `info` is a tuple of info dicts
            raise NotImplementedError("Vectorised environments not yet implemented")


    @abstractmethod
    def reset(self):
        """Resets the env and returns the state"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Takes an action in the env, and returns the next state, reward, done and info"""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Method to close and clean up the env"""
        raise NotImplementedError
