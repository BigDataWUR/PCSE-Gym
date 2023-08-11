import gymnasium as gym

from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_model_kwargs
import pcse_gym.utils.defaults as defaults


def get_crop_features(pcse_env=1):
    if pcse_env:
        crop_features = defaults.get_wofost_default_crop_features()
    else:
        crop_features = defaults.get_lintul_default_crop_features()
    return crop_features


def get_action_space(nitrogen_levels=7):
    space_return = gym.spaces.Discrete(nitrogen_levels)
    return space_return


def initialize_env(pcse_env=1, crop_features=get_crop_features(pcse_env=1),
                   costs_nitrogen=10, reward='DEF', nitrogen_levels=7, action_multiplier=1.0,
                   years=defaults.get_default_train_years(), locations=defaults.get_default_location()):
    action_space = get_action_space(nitrogen_levels=nitrogen_levels)
    env_return = WinterWheat(crop_features=crop_features,
                             costs_nitrogen=costs_nitrogen,
                             years=years,
                             locations=locations,
                             action_space=action_space,
                             action_multiplier=action_multiplier,
                             reward=reward,
                             **get_model_kwargs(pcse_env))

    return env_return


def initialize_env_no_baseline():
    return initialize_env(reward='GRO', action_multiplier=2.0)
