import os
from tqdm import tqdm
from collections import OrderedDict

import pandas as pd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
import pcse
from collections import defaultdict
import pcse_gym.envs.common_env
from pcse_gym.utils.defaults import *
from .rewards import *


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


def get_policy_kwargs(crop_features=get_wofost_default_crop_features(),
                      weather_features=get_default_weather_features(),
                      action_features=get_default_action_features(),
                      n_timesteps=7):
    # Integration with BaseModel from Stable Baselines3
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(n_timeseries=len(weather_features),
                                       n_scalars=len(crop_features) + len(action_features),
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
        reward_var='WSO'
    )
    return lintul_kwargs


def get_pcse_model(kwargs):
    if kwargs == 0:
        return get_lintul_kwargs()
    elif kwargs == 1:
        return get_wofost_kwargs()
    else:
        raise Exception("Choose 0 or 1 for the environment")


class StableBaselinesWrapper(pcse_gym.envs.common_env.PCSEEnv):
    """
    Establishes compatibility with Stable Baselines3

    :param action_multiplier: conversion factor to map output node to g/m2 of nitrogen
        action_space=gym.spaces.Discrete(3), action_multiplier=2.0 gives {0, 2.0, 4.0}
        action_space=gym.spaces.Box(0, np.inf, shape=(1,), action_multiplier=1.0 gives 1.0*x
    """

    def __init__(self, crop_features=get_wofost_default_crop_features(),
                 weather_features=get_default_weather_features(),
                 action_features=get_default_action_features(), costs_nitrogen=10.0, timestep=7,
                 years=None, location=None, seed=0, action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                 action_multiplier=1.0, *args, **kwargs):
        self.costs_nitrogen = costs_nitrogen
        self.crop_features = crop_features
        self.po_features = kwargs.get('po_features')
        self.weather_features = weather_features
        self.action_features = action_features
        super().__init__(timestep=timestep, years=years, location=location, *args, **kwargs)
        self.action_space = action_space
        self.action_multiplier = action_multiplier
        self.reward_var = kwargs.get('reward_var', "TWSO")

        self.rewards = Rewards(self.reward_var, self.timestep)

        self.index_feature = OrderedDict()
        super().reset(seed=seed)

    def _get_observation_space(self):
        nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self.timestep
        return gym.spaces.Box(0, np.inf, shape=(nvars,))

    def _apply_action(self, action):
        amount = action * self.action_multiplier
        self._model._send_signal(signal=pcse.signals.apply_n, N_amount=amount * 10, N_recovery=0.7,
                                 amount=amount, recovery=0.7)

    def _get_reward(self):
        return super()._get_reward(var=self.reward_var)

    def step(self, action):
        """
        Computes customized reward and populates info
        """
        if isinstance(action, np.ndarray):
            act, measure = action[0], action[1:]

        obs, _, terminated, truncated, info = super().step(action)
        output = self.model.get_output()

        benefits = self.rewards.storage_organ_growth(output)

        amount = action * self.action_multiplier
        costs = self.costs_nitrogen * amount
        reward = benefits - costs

        observation = self._observation(obs)

        crop_info = pd.DataFrame(output).set_index("day").fillna(value=np.nan)
        days = [day['day'] for day in output]
        weather_data = [self._weather_data_provider(day) for day in days]
        weather_variables = self._weather_variables
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
        info = {**pd.concat([crop_info, weather_info], axis=1, join="inner").to_dict()}

        if 'action' not in info.keys():
            info['action'] = {}
        if 'measure' not in info.keys():
            info['measure'] = {}
        if isinstance(action, np.ndarray):
            info['action'][output[-1 - self.timestep]['day']] = action[0]
            info['measure'][output[-1 - self.timestep]['day']] = measure
        else:
            info['action'][output[-1 - self.timestep]['day']] = action
        if 'fertilizer' not in info.keys():
            info['fertilizer'] = {}
        if isinstance(action, np.ndarray):
            info['fertilizer'][output[-1 - self.timestep]['day']] = amount[0]
        else:
            info['fertilizer'][output[-1 - self.timestep]['day']] = amount
        if 'reward' not in info.keys():
            info['reward'] = {}
        info['reward'][self.date] = reward
        obs['actions'] = {'cumulative_nitrogen': sum(info['fertilizer'].values())}
        obs['actions'] = {'cumulative_measurement': sum(info['measure'].values())}
        if 'indexes' not in info.keys():
            info['indexes'] = {}
            if self.index_feature:
                info['indexes'] = self.index_feature

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
        index_feature = OrderedDict()
        for i, feature in enumerate(self.crop_features):
            obs[i] = observation['crop_model'][feature][-1]
            if feature not in index_feature and not flag:
                if feature in self.po_features:
                    index_feature[feature] = i
                    if len(index_feature.keys()) == len(self.po_features):
                        self.index_feature = index_feature
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
    def location(self):
        return self._location

    @property
    def timestep(self):
        return self._timestep


class ZeroNitrogenEnvStorage:
    """
    Container to store results from zero nitrogen policy (for re-use)
    """

    def __init__(self, env_baseline, years, locations):
        self.results = {}
        self.run_through(env_baseline, years, locations)

    def run_episode(self, env, year, location):
        env._agro_management = pcse_gym.envs.common_env.replace_years(env._agro_management, year)
        env._location = location
        env._weather_data_provider = pcse_gym.envs.common_env.get_weather_data_provider(location)
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

    def get_episode_output(self, env, key, year, location):
        if key not in self.results.keys():
            results = self.run_episode(env, year, location)
            self.results[key] = results

    def run_through(self, env, years, locations):
        print('creating zero nitrogen results...')
        for year in tqdm(years):
            for location in locations:
                key = f'{year}-{location}'
                self.get_episode_output(env, key, year, location)

    @property
    def get_result(self):
        return self.results
