from tac.env.base import BaseEnv
import numpy as np
import pydash as ps
from tac.env.wrapper import make_gym_env


class OpenAIEnv(BaseEnv):

    def __init__(self, spec):
        super().__init__(spec)
        # TODO: try_register_env(spec)
        self.seed = ps.get(spec, "random_seed")
        # TODO: episode_life = utils.general.in_train_lab_mode()
        if self.is_venv:
            raise NotImplementedError("Vectorised environments not yet supported")
        else:
            self.u_env = make_gym_env(
                name=self.name,
                # num_envs=self.num_envs,
                seed=self.seed,
                # frame_op=self.frame_op,
                # frame_op_len=self.frame_op_len,
                # image_downsize=self.image_downsize,
                # reward_scale=self.reward_scale,
                # normalise_state=self.normalise_state,
                # episode_life=self.episode_life,
            )
        # TODO: check next line
        if self.name.startswith("Unity"):
            raise NotImplementedError("Unity environments not yet supported")
        self._set_attr_from_u_env(self.u_env)
        # TODO: check next line
        self.max_t = self.max_t or self.u_env.spec.max_episode_steps
        assert self.max_t is not None, "max_t not specified in spec or environment"

    def seed(self, seed):
        self.u_env.seed(seed)

    def reset(self):
        """Reset the environment and return the initial state"""
        # TODO: check done or terminated
        self.done = False
        state = self.u_env.reset()
        # if self.to_render:
        #     self.u_env.render()
        return state

    def step(self, action):
        """Take a step in the environment and return the next state, reward, and whether the episode is terminated"""
        if not self.is_discrete and self.action_dim == 1:   # Guard for continuous with action_dim 1, make array
            action = np.expand_dims(action, axis=-1)
        state, reward, terminated, truncated, info = self.u_env.step(action)
        # if self.to_render:
        #     self.u_env.render()
        # if not self.is_venv and self.clock.t > self.max_t:
        if self.clock.t > self.max_t:
            terminated = True
        self.done = terminated
        return state, reward, terminated, truncated, info

    def close(self):
        """Close the environment"""
        self.u_env.close()
