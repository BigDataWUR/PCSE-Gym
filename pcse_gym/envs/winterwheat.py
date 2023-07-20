import datetime

import pcse_gym.envs.common_env
from .sb3 import *
from .constraints import MeasureOrNot
from pcse_gym.utils.defaults import *


class WinterWheat(gym.Env):
    """
    @Code by Michiel Kallenberg 2022

    Environment with two sub-environments:
        (1) environment for applying actions of RL agent
        (2) a baseline environment (e.g. with zero nitrogen policy) for computing relative reward
    Year and location of episode is randomly picked from years and locations through reset().
    """

    def __init__(self, crop_features=get_wofost_default_crop_features(),
                 action_features=get_default_action_features(),
                 weather_features=get_default_weather_features(),
                 seed=0, costs_nitrogen=None, timestep=7, years=None, locations=None,
                 action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                 action_multiplier=1.0, reward=None,
                 *args, **kwargs
                 ):
        self.crop_features = crop_features
        self.action_features = action_features
        self.weather_features = weather_features
        if 'ANE' in reward:
            self.costs_nitrogen = 5.0
        else:
            self.costs_nitrogen = costs_nitrogen
        self.years = [years] if isinstance(years, int) else years
        self.locations = [locations] if isinstance(locations, tuple) else locations
        self.action_multiplier = action_multiplier
        self.po_features = kwargs.get('po_features')
        self.action_space = action_space
        self._timestep = timestep
        self.reward_function = reward
        self.reward_var = kwargs.get('reward_var', "TWSO")

        if self.reward_function != 'GRO':
            self._env_baseline = StableBaselinesWrapper(crop_features=crop_features,
                                                        action_features=action_features,
                                                        weather_features=weather_features,
                                                        costs_nitrogen=costs_nitrogen,
                                                        timestep=timestep,
                                                        years=self.years[0], location=self.locations[0],
                                                        action_space=action_space,
                                                        action_multiplier=action_multiplier,
                                                        seed=seed,
                                                        *args, **kwargs)
        self._env = StableBaselinesWrapper(crop_features=crop_features,
                                           action_features=action_features,
                                           weather_features=weather_features,
                                           costs_nitrogen=costs_nitrogen,
                                           timestep=timestep,
                                           years=self.years[0], location=self.locations[0],
                                           action_space=action_space,
                                           action_multiplier=action_multiplier,
                                           seed=seed,
                                           *args, **kwargs)
        self.observation_space = self._get_observation_space()

        if self.reward_function != 'GRO':
            self.zero_nitrogen_env_storage = ZeroNitrogenEnvStorage(self._env_baseline, self.years, self.locations)
            self.rewards = Rewards(self.reward_var, self.timestep,
                                   zero_container=self.zero_nitrogen_env_storage)
        else:
            self.rewards = Rewards(self.reward_var, self.timestep)

        self.__measure = MeasureOrNot(self.sb3_env)

        super().reset(seed=seed)

    def _get_observation_space(self):
        nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self.timestep
        return gym.spaces.Box(0, np.inf, shape=(nvars,))

    def step(self, action):
        """
        Computes customized reward and populates info
        """

        obs, _, terminated, truncated, info = self._env.step(action)

        if isinstance(action, np.ndarray):
            act, measure = action[0], action[1:]

        output = self.sb3_env.model.get_output()
        amount = act * self.action_multiplier
        reward, growth = self.get_reward_func(output, amount)
        obs, cost = self.measure_features.measure_act(obs, measure)
        measurement_cost = sum(cost)
        reward -= measurement_cost

        if 'reward' not in info.keys(): info['reward'] = {}
        info['reward'][self.date] = reward
        if 'growth' not in info.keys(): info['growth'] = {}
        info['growth'][self.date] = growth

        return obs, reward, terminated, truncated, info

    def get_reward_func(self, output, amount):
        match self.reward_function:  # Needs python 3.10+
            case 'ANE':
                return self.rewards.ane_reward(output, amount)
            case 'DEF':
                return self.rewards.default_winterwheat_reward(output, amount)
            case 'GRO':
                return self.rewards.growth_reward(output, amount)
            case _:
                return self.rewards.default_winterwheat_reward(output, amount)

    def overwrite_year(self, year):
        self.years = year
        if self.reward_function != 'GRO':
            self._env_baseline._agro_management = pcse_gym.envs.common_env.replace_years(
                self._env_baseline._agro_management, year)
        self._env._agro_management = pcse_gym.envs.common_env.replace_years(self._env._agro_management, year)

    def set_location(self, location):
        if self.reward_function != 'GRO':
            self._env_baseline._location = location
            self._env_baseline._weather_data_provider = pcse_gym.envs.common_env.get_weather_data_provider(location)
        self._env._location = location
        self._env._weather_data_provider = pcse_gym.envs.common_env.get_weather_data_provider(location)

    def reset(self, seed=None, options=()):
        if isinstance(self.years, list):
            year = self.np_random.choice(self.years)
            if self.reward_function != 'GRO':
                self._env_baseline._agro_management = pcse_gym.envs.common_env.replace_years(
                    self._env_baseline._agro_management, year)
            self._env._agro_management = pcse_gym.envs.common_env.replace_years(self._env._agro_management, year)

        if isinstance(self.locations, list):
            location = self.locations[self.np_random.choice(len(self.locations), 1)[0]]
            self.set_location(location)
        if self.reward_function != 'GRO':
            self._env_baseline.reset(seed=seed)
        obs = self._env.reset(seed=seed)

        # TODO: check whether info should/could be filled
        info = {}

        return obs, info

    def render(self, mode="human"):
        pass

    @property
    def measure_features(self):
        return self.__measure

    @property
    def sb3_env(self):
        return self._env

    @property
    def baseline_env(self):
        return self._env_baseline

    @property
    def date(self) -> datetime.date:
        return self.sb3_env.model.day

    @property
    def loc(self) -> datetime.date:
        return self.sb3_env.location

    @property
    def timestep(self):
        return self._timestep


