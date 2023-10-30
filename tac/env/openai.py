from tac.env.base import BaseEnv
import gymnasium as gym
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
            # self.env.seed(self.seed)
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
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
