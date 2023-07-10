import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper

import envs.winterwheat
import pcse_gym.envs.winterwheat


def mask_fertilization_actions(env: pcse_gym.envs.winterwheat.WinterWheat) -> np.ndarray:

    return env.valid_action_mask()


class ActionLimiter(ActionWrapper):
    def __init__(self, env, action_limit):
        super(ActionLimiter, self).__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.counter = 0
        self.action_limit = action_limit

    def action(self, action):
        if action != 0: #if there's an action, increase the counter
            self.counter += 1
        if self.counter > self.action_limit: #return 0 if the action exceeds limit
            action = 0
        return action

    def reset(self, **kwargs):
        self.counter = 0
        return self.env.reset(**kwargs)


class MeasureOrNot(ObservationWrapper):
    pass


