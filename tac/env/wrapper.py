import gymnasium as gym
import numpy as np


class TrackReward(gym.Wrapper):
    """TODO: Docstring (after unit testing this class and understanding all aspects)"""

    def __init__(self, env):
        # TODO: slightly different to SLM-Lab. Check this
        super().__init__(env)
        self.tracked_reward = 0
        self.total_reward = 0

    def step(self, action):
        # TODO: this is different to SLM-Lab - gymnasium splits old "done" into "terminated" and "truncated". Check
        #  implications on upstream code
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.tracked_reward += reward
        # Fix shape by inferring from reward
        if np.isscalar(self.total_reward) and not np.isscalar(reward):
            # TODO: explain this
            self.total_reward = np.full_like(reward, self.total_reward)
        # Use self.was_real_done from EpisodicLifeEnv, or plain old done. info.get(..., False) returns False if the
        # key is not found
        # TODO: terminated rather than (in SLM-Lab) done
        real_terminated = info.get("was_real_done", False) or terminated
        not_real_terminated = (1 - real_terminated)
        # If isnan and at done, reset total_reward from nan to 0 so it can be updated with the tracked_reward
        # TODO: understand all if statements, and streamline if possible
        if np.isnan(self.total_reward).any():
            if np.isscalar(self.total_reward):
                if np.isnan(self.total_reward) and real_terminated:
                    self.total_reward = 0.0
            else:
                replace_locs = np.logical_and(np.isnan(self.total_reward), real_terminated)
                self.total_reward[replace_locs] = 0.0
        # Update total reward
        self.total_reward = self.total_reward * not_real_terminated + self.tracked_reward * real_terminated
        # Reset tracked reward
        self.tracked_reward = self.tracked_reward * not_real_terminated
        info.update({"total_reward": self.total_reward})
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.tracked_reward = 0
        return self.env.reset(**kwargs)


def make_gym_env(
        name,
        seed=None,
        frame_op=None,
        frame_op_len=None,
        image_downsize=None,
        reward_scale=None,
        normalise_state=False,
        episode_life=True,
):
    """
    General method to create any Gym env from a name.

    TODO: check below:
    :param name: name of the gym env to make
    :param seed: random seed to use for the env
    :param frame_op: frame operation to apply to the env
    :param frame_op_len: length of the frame operation to apply to the env
    :param image_downsize: size to downsize the image to
    :param reward_scale: reward scale to apply to the env
    :param normalise_state: whether to normalise the state
    :param episode_life: whether to terminate episodes when a life is lost
    :return: gym env
    """

    env = gym.make(name)
    observation, info = env.reset(seed=seed, options={})

    # if seed is not None:
    #     # TODO: check linting error in line below
    #     env.seed(seed)
    # if "NoFrameskip" in env.spec:    # Test for Atari env
    #     raise NotImplementedError("Atari environments not yet supported")
    # elif len(env.observation_space.shape) == 3:    # Test for image env
    if len(env.observation_space.shape) == 3:    # Test for image env
        raise NotImplementedError("Image environments not yet supported")
    else:    # Vector state env
        if normalise_state:
            raise NotImplementedError("State normalisation not yet supported")
        if frame_op is not None:
            raise NotImplementedError("Frame operations not yet supported")
    env = TrackReward(env)    # Auto-track total reward
    if reward_scale is not None:
        raise NotImplementedError("Reward scaling not yet supported")
    return env
