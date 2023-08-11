import os
from collections import defaultdict
import gymnasium as gym
import pandas as pd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import pcse
import numpy as np

import pcse_gym.envs.common_env as common_env
import pcse_gym.utils.defaults as defaults
import pcse_gym.utils.process_pcse_output as process_pcse
from .rewards import Rewards


def to_weather_info(days, weather_data, weather_variables):
    weather_observation = []
    for i, d in enumerate(days):
        def def_value():
            return 0
        w = defaultdict(def_value)
        w['day'] = d
        for var in weather_variables:
            w[var] = getattr(weather_data[i], var)
        weather_observation.append(w)
    weather_info = pd.DataFrame(weather_observation).set_index("day").fillna(value=np.nan)
    return weather_info


def update_info(inf, key, date, value):
    if key not in inf.keys():
        inf[key] = {}
    inf[key][date] = value
    return inf


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Processes input features: average pool timeseries (weather) and concat with scalars (crop features)
    """

    def __init__(self, observation_space: gym.spaces.Box, n_timeseries, n_scalars, n_timesteps=7):
        self.n_timeseries = n_timeseries
        self.n_scalars = n_scalars
        self.n_timesteps = n_timesteps
        super(CustomFeatureExtractor, self).__init__(gym.spaces.Box(0, np.inf, shape=(n_timeseries + n_scalars,)),
                                                     features_dim=n_timeseries + n_scalars)

        self.avg_timeseries = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.n_timesteps)
        )

    def forward(self, observations) -> th.Tensor:
        # Returns a torch tensor in a format compatible with Stable Baselines3
        batch_size = observations.shape[0]
        scalars, timeseries = observations[:, 0:self.n_scalars], \
                              observations[:, self.n_scalars:]
        reshaped = timeseries.reshape(batch_size, self.n_timesteps, self.n_timeseries).permute(0, 2, 1)
        x1 = self.avg_timeseries(reshaped)
        x1 = th.squeeze(x1, 2)
        x = th.cat((x1, scalars), dim=1)
        return x


def get_policy_kwargs(n_crop_features=len(defaults.get_wofost_default_crop_features()),
                      n_weather_features=len(defaults.get_default_weather_features()),
                      n_action_features=len(defaults.get_default_action_features()),
                      n_timesteps=7):
    # Integration with BaseModel from Stable Baselines3
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(n_timeseries=n_weather_features,
                                       n_scalars=n_crop_features + n_action_features,
                                       n_timesteps=n_timesteps),
    )
    return policy_kwargs


def get_config_dir():
    from pathlib import Path
    config_dir = os.path.join(Path(os.path.realpath(__file__)).parents[1], 'envs', 'configs')
    return config_dir


def get_wofost_kwargs(config_dir=get_config_dir()):
    wofost_kwargs = dict(
        model_config=os.path.join(config_dir, 'Wofost81_NWLP_FD.conf'),
        agro_config=os.path.join(config_dir, 'agro', 'wheat_cropcalendar.yaml'),
        crop_parameters=pcse.fileinput.YAMLCropDataProvider(fpath=os.path.join(config_dir, 'crop'), force_reload=True),
        site_parameters=pcse.util.WOFOST80SiteDataProvider(WAV=10, NAVAILI=10, PAVAILI=50, KAVAILI=100),
        soil_parameters=pcse.fileinput.CABOFileReader(os.path.join(config_dir, 'soil', 'ec3.CAB'))
    )
    return wofost_kwargs


def get_lintul_kwargs(config_dir=get_config_dir()):
    lintul_kwargs = dict(
        model_config=os.path.join(config_dir, 'Lintul3.conf'),
        agro_config=os.path.join(config_dir, 'agro', 'agromanagement_fertilization.yaml'),
        crop_parameters=os.path.join(config_dir, 'crop', 'lintul3_winterwheat.crop'),
        site_parameters=os.path.join(config_dir, 'site', 'lintul3_springwheat.site'),
        soil_parameters=os.path.join(config_dir, 'soil', 'lintul3_springwheat.soil'),
    )
    return lintul_kwargs


def get_model_kwargs(pcse_model):
    if pcse_model == 0:
        return get_lintul_kwargs()
    elif pcse_model == 1:
        return get_wofost_kwargs()
    else:
        raise Exception("Choose 0 or 1 for the environment")


class StableBaselinesWrapper(common_env.PCSEEnv):
    """
    Establishes compatibility with Stable Baselines3

    :param action_multiplier: conversion factor to map output node to g/m2 of nitrogen
        action_space=gym.spaces.Discrete(3), action_multiplier=2.0 gives {0, 2.0, 4.0}
        action_space=gym.spaces.Box(0, np.inf, shape=(1,), action_multiplier=1.0 gives 1.0*x
    """

    def __init__(self, crop_features=defaults.get_wofost_default_crop_features(),
                 weather_features=defaults.get_default_weather_features(),
                 action_features=defaults.get_default_action_features(), costs_nitrogen=10.0, timestep=7,
                 years=None, location=None, seed=0, action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                 action_multiplier=1.0, *args, **kwargs):
        self.costs_nitrogen = costs_nitrogen
        self.crop_features = crop_features
        self.weather_features = weather_features
        self.action_features = action_features
        super().__init__(timestep=timestep, years=years, location=location, *args, **kwargs)
        self.action_space = action_space
        self.action_multiplier = action_multiplier
        self.rewards = Rewards(kwargs.get('reward_var'), self.timestep, self.costs_nitrogen)
        super().reset(seed=seed)

    def _get_observation_space(self):
        nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self.timestep
        return gym.spaces.Box(0, np.inf, shape=(nvars,))

    def _apply_action(self, action):
        amount = action * self.action_multiplier
        recovery_rate = 0.7
        self._model._send_signal(signal=pcse.signals.apply_n, N_amount=amount * 10, N_recovery=recovery_rate,
                                 amount=amount, recovery=recovery_rate)

    def _get_reward(self):
        # Reward gets overwritten in step()
        return 0

    def step(self, action):
        """
        Computes customized reward and populates info
        """

        obs, _, terminated, truncated, _ = super().step(action)

        # populate observation
        observation = self._observation(obs)

        # populate reward
        pcse_output = self.model.get_output()
        amount = action * self.action_multiplier
        reward, growth = self.rewards.growth_storage_organ(pcse_output, amount)

        # populate info
        crop_info = pd.DataFrame(pcse_output).set_index("day").fillna(value=np.nan)
        days = [day['day'] for day in pcse_output]
        weather_data = [self._weather_data_provider(day) for day in days]
        weather_info = to_weather_info(days, weather_data, self._weather_variables)
        info = {**pd.concat([crop_info, weather_info], axis=1, join="inner").to_dict()}

        start_date = process_pcse.get_start_date(pcse_output, self.timestep)
        info = update_info(info, 'action', start_date, action)
        info = update_info(info, 'fertilizer', start_date, amount)
        info = update_info(info, 'reward', self.date, reward)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, return_info=False, options=None):

        obs = super().reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs['actions'] = {'cumulative_nitrogen': 0.0}
        return self._observation(obs, flag=True)

    def _observation(self, observation, flag=False):
        """
        Converts observation into np array to facilitate integration with Stable Baseline3
        """
        obs = np.zeros(self.observation_space.shape)

        if isinstance(observation, tuple):
            observation = observation[0]
        for i, feature in enumerate(self.crop_features):
            obs[i] = observation['crop_model'][feature][-1]

        for i, feature in enumerate(self.action_features):
            j = len(self.crop_features) + i
            obs[j] = observation['actions'][feature]
        for d in range(self.timestep):
            for i, feature in enumerate(self.weather_features):
                j = d * len(self.weather_features) + len(self.crop_features) + len(self.action_features) + i
                obs[j] = observation['weather'][feature][d]
        return obs

    @property
    def model(self):
        return self._model

    @property
    def loc(self):
        return self._location

    @loc.setter
    def loc(self, location):
        self._location = location

    @property
    def timestep(self):
        return self._timestep

    @property
    def agro_management(self):
        return self._agro_management

    @agro_management.setter
    def agro_management(self, agro):
        self._agro_management = agro

    @property
    def weather_data_provider(self):
        return self._weather_data_provider

    @weather_data_provider.setter
    def weather_data_provider(self, weather):
        self._weather_data_provider = weather


class ZeroNitrogenEnvStorage:
    """
    Container to store results from zero nitrogen policy (for re-use)
    """

    def __init__(self):
        self.results = {}

    def run_episode(self, env):
        env.reset()
        terminated, truncated = False, False
        infos_this_episode = []
        while not terminated or truncated:
            _, _, terminated, truncated, info = env.step(0)
            infos_this_episode.append(info)
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        return episode_info

    def get_key(self, env):
        year = env.date.year
        location = env.loc
        return f'{year}-{location}'

    def get_episode_output(self, env):
        key = self.get_key(env)
        if key not in self.results.keys():
            results = self.run_episode(env)
            self.results[key] = results
        return self.results[key]

    @property
    def get_result(self):
        return self.results