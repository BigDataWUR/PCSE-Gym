import gym
import datetime
import numpy as np
import pandas as pd
import pcse
from collections import defaultdict
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pcse_gym.environment.env


def get_default_crop_features():
    return ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN",
            "TIRRIG", "TNSOIL", "TRAIN", "TRANRF", "TRUNOF", "TAGBM",
            "TTRAN", "WC", "WLVD", "WLVG", "WRT", "WSO", "WST"]


def get_default_weather_features():
    return ["IRRAD", "TMIN", "TMAX", "VAP", "RAIN", "E0", "ES0", "ET0", "WIND"]


def get_default_action_features():
    return []
    #return ["cumulative_nitrogen"]


def get_default_train_years():
    return [2014, 2015, 2016]


def get_default_test_years():
    return [2018, 2019, 2020]


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_timeseries, n_scalars, n_timesteps = 7):
        self.n_timeseries = n_timeseries
        self.n_scalars = n_scalars
        self.n_timesteps = n_timesteps
        super(CustomFeatureExtractor, self).__init__(gym.spaces.Box(0, np.inf, shape=(n_timeseries + n_scalars,)),
                                                     features_dim=n_timeseries + n_scalars)

        self.avg_timeseries = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.n_timesteps)
        )

    def forward(self, observations) -> th.Tensor:
        batch_size = observations.shape[0]
        scalars, timeseries = observations[:, 0:self.n_scalars], \
                              observations[:, self.n_scalars:]
        reshaped = timeseries.reshape(batch_size, self.n_timesteps, self.n_timeseries).permute(0,2,1)
        x1 = self.avg_timeseries(reshaped)
        x1 = th.squeeze(x1,2)
        x = th.cat((x1, scalars), dim=1)
        return x


def get_policy_kwargs(crop_features=get_default_crop_features(),
                      weather_features=get_default_weather_features(),
                      action_features=get_default_action_features(),
                      n_timesteps=7):
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(n_timeseries=len(weather_features),
                                       n_scalars=len(crop_features) + len(action_features),
                                       n_timesteps=n_timesteps),
    )
    return policy_kwargs


class StableBaselinesWrapper(pcse_gym.environment.env.PCSEEnv):
    def __init__(self, crop_features=get_default_crop_features(), weather_features=get_default_weather_features(),
                 action_features=get_default_action_features(), costs_nitrogen=5.0, timestep=7,
                 years=None, action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                 action_multiplier=1.0):
        self.costs_nitrogen = costs_nitrogen
        self.crop_features = crop_features
        self.weather_features = weather_features
        self.action_features = action_features
        super().__init__(timestep=timestep, years=years)
        self.action_space = action_space
        self.action_multiplier = action_multiplier

    def _get_observation_space(self):
        nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self._timestep
        return gym.spaces.Box(0, np.inf, shape=(nvars,))

    def _apply_action(self, action):
        amount = action * self.action_multiplier
        self._model._send_signal(signal=pcse.signals.apply_n, amount=amount, recovery=0.7)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        obs, _, done, info = super().step(action)
        output = self._model.get_output()
        last_index_previous_state = (np.ceil(len(output) / self._timestep).astype('int') - 1) * self._timestep - 1
        wso_start = output[last_index_previous_state]['WSO']
        wso_finish = output[-1]['WSO']
        benefits = wso_finish - wso_start
        amount = action * self.action_multiplier
        costs = self.costs_nitrogen * amount
        reward = benefits - costs

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
        info['action'][output[-1 - self._timestep]['day']] = action
        if 'fertilizer' not in info.keys():
            info['fertilizer'] = {}
        info['fertilizer'][output[-1 - self._timestep]['day']] = amount
        if 'reward' not in info.keys():
            info['reward'] = {}
        info['reward'][self.date] = reward
        obs['actions'] = {'cumulative_nitrogen': sum(info['fertilizer'].values())}
        return self._observation(obs), reward, done, info

    def reset(self):
        obs = super().reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs['actions'] = {'cumulative_nitrogen': 0.0}
        return self._observation(obs)

    def _observation(self, observation):
        obs = np.zeros(self.observation_space.shape)

        if isinstance(observation, tuple):
            observation = observation[0]

        for i, feature in enumerate(self.crop_features):
            obs[i] = observation['crop_model'][feature][-1]
        for i, feature in enumerate(self.action_features):
            j = len(self.crop_features) + i
            obs[j] = observation['actions'][feature]
        for d in range(self._timestep):
            for i, feature in enumerate(self.weather_features):
                j = d * len(self.weather_features) + len(self.crop_features) + len(self.action_features) + i
                obs[j] = observation['weather'][feature][d]
        return obs


class ZeroNitrogenEnvStorage():
    def __init__(self):
        self.results = {}

    def run_episode(self, env):
        env.reset()
        done = False
        infos_this_episode = []
        while not done:
            _, _, done, info = env.step(0)
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
        return year

    def get_episode_output(self, env):
        key = self.get_key(env)
        if key not in self.results.keys():
            results = self.run_episode(env)
            self.results[key] = results
        return self.results[key]


class ReferenceEnv(gym.Env):
    def __init__(self, crop_features=get_default_crop_features(),
                 action_features=get_default_action_features(),
                 weather_features=get_default_weather_features(),
                 seed=0, costs_nitrogen=10.0, timestep=7, years=None,
                 action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                 action_multiplier=1.0):
        self.crop_features = crop_features
        self.action_features = action_features
        self.weather_features = weather_features
        self.costs_nitrogen = costs_nitrogen
        self.years = years
        self.action_multiplier = action_multiplier
        self.action_space = action_space
        self._timestep = timestep
        self._env_baseline = StableBaselinesWrapper(crop_features=crop_features,
                                                    action_features=action_features,
                                                    weather_features=weather_features,
                                                    costs_nitrogen=costs_nitrogen,
                                                    timestep=timestep, years=[years][0], action_space=action_space,
                                                    action_multiplier=action_multiplier)
        self._env = StableBaselinesWrapper(crop_features=crop_features,
                                           action_features=action_features,
                                           weather_features=weather_features,
                                           costs_nitrogen=costs_nitrogen,
                                           timestep=timestep, years=[years][0], action_space=action_space,
                                           action_multiplier=action_multiplier)
        self.observation_space = self._get_observation_space()
        self.zero_nitrogen_env_storage = ZeroNitrogenEnvStorage()

        self.seed(seed)

    def _get_observation_space(self):
        nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self._timestep
        return gym.spaces.Box(0, np.inf, shape=(nvars,))

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        obs, _, done, info = self._env.step(action)

        output = self._env._model.get_output()
        last_index_previous_state = (np.ceil(len(output) / self._timestep) - 1).astype('int') * self._timestep - 1
        growth = output[-1]['WSO'] - output[last_index_previous_state]['WSO']

        current_date = output[-1]['day']
        previous_date = output[last_index_previous_state]['day']

        self.zero_nitrogen_env_storage.get_episode_output(self._env_baseline)
        wso_current = self.zero_nitrogen_env_storage.results[current_date.year]['WSO'][current_date]
        wso_previous = self.zero_nitrogen_env_storage.results[current_date.year]['WSO'][previous_date]
        growth_baseline = wso_current - wso_previous

        benefits = growth - growth_baseline
        amount = action * self.action_multiplier
        costs = self.costs_nitrogen * amount
        reward = benefits - costs
        if 'reward' not in info.keys(): info['reward'] = {}
        info['reward'][self.date] = reward

        if 'growth' not in info.keys(): info['growth'] = {}
        info['growth'][self.date] = growth

        return obs, reward, done, info

    def seed(self, seed=None):
        """
        fix the random seed
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def overwrite_year(self, year):
        self.years = year
        self._env_baseline._agro_management = pcse_gym.environment.env.replace_years(
            self._env_baseline._agro_management, year)
        self._env._agro_management = pcse_gym.environment.env.replace_years(self._env._agro_management, year)

    def reset(self):
        if isinstance(self.years, list):
            year = self.np_random.choice(self.years)
            self._env_baseline._agro_management = pcse_gym.environment.env.replace_years(
                self._env_baseline._agro_management, year)
            self._env._agro_management = pcse_gym.environment.env.replace_years(self._env._agro_management, year)

        self._env_baseline.reset()
        obs = self._env.reset()

        return obs

    def render(self, mode="human"):
        pass

    @property
    def date(self) -> datetime.date:
        return self._env._model.day
