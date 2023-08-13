import gymnasium as gym

from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_model_kwargs
import pcse_gym.utils.defaults as defaults


def get_po_features(pcse_env=1):
    if pcse_env:
        po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal']
    else:
        po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
    return po_features


def get_action_space(nitrogen_levels=7, po_features=[]):
    if po_features:
        a_shape = [nitrogen_levels] + [2] * len(po_features)
        space_return = gym.spaces.MultiDiscrete(a_shape)
    else:
        space_return = gym.spaces.Discrete(nitrogen_levels)
    return space_return


def initialize_env(pcse_env=1, po_features=[], crop_features=defaults.get_default_crop_features(pcse_env=1),
                   costs_nitrogen=10, reward='DEF', nitrogen_levels=7, action_multiplier=1.0, add_random=False,
                   years=defaults.get_default_train_years(), locations=defaults.get_default_location(), args_vrr=False):
    if add_random:
        po_features.append('random'), crop_features.append('random')
    action_space = get_action_space(nitrogen_levels=nitrogen_levels, po_features=po_features)
    kwargs = dict(po_features=po_features, args_measure=po_features is not None, args_vrr=args_vrr)
    env_return = WinterWheat(crop_features=crop_features,
                             costs_nitrogen=costs_nitrogen,
                             years=years,
                             locations=locations,
                             action_space=action_space,
                             action_multiplier=action_multiplier,
                             reward=reward,
                             **get_model_kwargs(pcse_env),
                             **kwargs)

    return env_return


def initialize_env_po():
    return initialize_env(po_features=get_po_features())


def initialize_env_no_baseline():
    return initialize_env(reward='GRO', action_multiplier=2.0)


def initialize_env_random():
    return initialize_env(po_features=get_po_features(), add_random=True)


def initialize_env_rr():
    return initialize_env(reward='GRO', args_vrr=True)
