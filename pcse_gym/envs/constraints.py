import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pcse_gym.envs.winterwheat


def mask_fertilization_actions(env: pcse_gym.envs.winterwheat.WinterWheat) -> np.ndarray:

    return env.valid_action_mask()
