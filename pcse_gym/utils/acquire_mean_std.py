import numpy as np

import gymnasium as gym
from tqdm import tqdm

import pcse_gym.utils.defaults as defaults
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_model_kwargs


def gather_mean_and_std(env, locations, years, episodes=30):
    """
    Gather observation statistics using random policy.
    """

    all_observations = []

    for _ in tqdm(range(episodes)):
        for year in years:
            for loc in locations:
                env.overwrite_year(year)
                env.overwrite_location(loc)
                obs = env.reset()
                while True:
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    all_observations.append(obs)
                    action = env.action_space.sample()  # Random policy
                    obs, _, terminate, _, _ = env.step(action)
                    if terminate:
                        break

    all_observations = np.array(all_observations)

    means = all_observations.mean(axis=0)
    std_devs = all_observations.std(axis=0)

    return means, std_devs


years = [*range(1985, 2022)]
# NL
locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
# LT
# locations = [(55.0, 23.5), (55.0, 24.0), (55.5, 23.5)]

no_weather = False

env = WinterWheat(crop_features=defaults.get_default_crop_features(pcse_env=1, minimal=True),
                  weather_features=defaults.get_default_weather_features(),
                  action_features=defaults.get_default_action_features(),
                  costs_nitrogen=10,
                  years=years,
                  locations=locations,
                  action_space=gym.spaces.Discrete(7),
                  action_multiplier=1,
                  reward='GRO',
                  args_measure=False,
                  po_features=[],
                  no_weather=no_weather,
                  **get_model_kwargs(1, locations, start_type='sowing'))

means, std_devs = gather_mean_and_std(env, locations, years, 100)

print(f"means: {means}")
print(f"mean DVS: {means[0]}")
print(f"mean TAGP: {means[1]}")
print(f"means LAI: {means[2]}")
print(f"means NuptakeTotal: {means[3]}")
print(f"means NAVAIL: {means[4]}")
print(f"means SM: {means[5]}")
if not no_weather:
    print(f"means weather {means[6:]}")

print(f"std: {std_devs}")
print(f"std DVS: {std_devs[0]}")
print(f"std TAGP: {std_devs[1]}")
print(f"std LAI: {std_devs[2]}")
print(f"std NuptakeTotal: {std_devs[3]}")
print(f"std NAVAIL: {std_devs[4]}")
print(f"std SM: {std_devs[5]}")
if not no_weather:
    print(f"std weather {means[6:]}")
