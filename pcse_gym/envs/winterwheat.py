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
        self.costs_nitrogen = costs_nitrogen
        self.years = [years] if isinstance(years, int) else years
        self.locations = [locations] if isinstance(locations, tuple) else locations
        self.action_multiplier = action_multiplier
        self.po_features = kwargs.get('po_features', [])
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
        self.zero_nitrogen_env_storage = ZeroNitrogenEnvStorage()
        self.rewards = Rewards(self.reward_var, self.timestep)

        if self.po_features:
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

        output = self.sb3_env.model.get_output()

        obs, reward, growth = self.process_output(action, output, obs)

        if 'reward' not in info.keys(): info['reward'] = {}
        info['reward'][self.date] = reward
        if 'growth' not in info.keys(): info['growth'] = {}
        info['growth'][self.date] = growth

        return obs, reward, terminated, truncated, info

    def process_output(self, action, output, obs):

        if self.po_features and isinstance(action, np.ndarray) and action.dtype != np.float32:
            if isinstance(action, np.ndarray):
                action, measure = action[0], action[1:]
            amount = action * self.action_multiplier
            reward, growth = self.get_baseline_env(output, amount)
            obs, cost = self.measure_features.measure_act(obs, measure)
            measurement_cost = sum(cost)
            reward -= measurement_cost
            return obs, reward, growth
        else:
            if isinstance(action, np.ndarray):
                action = action.item()
            amount = action * self.action_multiplier
            reward, growth = self.get_baseline_env(output, amount)
            return obs, reward, growth

    def get_baseline_env(self, output, amount):
        if self.reward_function != 'GRO':
            zero_nitrogen_results = self.zero_nitrogen_env_storage.get_episode_output(self.baseline_env)

            # convert zero_nitrogen_results to pcse_output
            output_baseline = []
            for (k, v) in zero_nitrogen_results[self.reward_var].items():
                if k <= output[-1]['day']:
                    filtered_dict = {'day': k, self.reward_var: v}
                    output_baseline.append(filtered_dict)

            reward, growth = self.get_reward_func(output, amount, output_baseline)
        else:
            reward, growth = self.get_reward_func(output, amount)
        return reward, growth

    def get_reward_func(self, output, amount, output_baseline=None):
        match self.reward_function:  # Needs python 3.10+
            case 'ANE':
                return self.rewards.ane_reward(output, output_baseline, amount)
            case 'DEF':
                return self.rewards.default_winterwheat_reward(output, output_baseline, amount)
            case 'GRO':
                return self.rewards.growth_reward(output, amount)
            case _:
                return self.rewards.default_winterwheat_reward(output, output_baseline, amount)

    def overwrite_year(self, year):
        self.years = year
        if self.reward_function != 'GRO':
            self.baseline_env.agro_management = pcse_gym.envs.common_env.replace_years(
                self.baseline_env.agro_management, year)
        self.sb3_env.agro_management = pcse_gym.envs.common_env.replace_years(self.sb3_env.agro_management, year)

    def set_location(self, location):
        if self.reward_function != 'GRO':
            self.baseline_env.loc = location
            self.baseline_env.weather_data_provider = pcse_gym.envs.common_env.get_weather_data_provider(location)
        self.sb3_env.loc = location
        self.sb3_env.weather_data_provider = pcse_gym.envs.common_env.get_weather_data_provider(location)

    def overwrite_location(self, location):
        self.locations = location
        self.set_location(location)

    def reset(self, seed=None, options=()):
        if isinstance(self.years, list):
            year = self.np_random.choice(self.years)
            if self.reward_function != 'GRO':
                self.baseline_env.agro_management = pcse_gym.envs.common_env.replace_years(
                    self.baseline_env.agro_management, year)
            self.sb3_env.agro_management = pcse_gym.envs.common_env.replace_years(self.sb3_env.agro_management, year)

        if isinstance(self.locations, list):
            location = self.locations[self.np_random.choice(len(self.locations), 1)[0]]
            self.set_location(location)
        if self.reward_function != 'GRO':
            self.baseline_env.reset(seed=seed)
        obs = self.sb3_env.reset(seed=seed)

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
        return self.sb3_env.loc

    @property
    def timestep(self):
        return self._timestep
