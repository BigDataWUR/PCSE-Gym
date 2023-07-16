import datetime

from collections import OrderedDict

import pcse_gym.envs.common_env
from .sb3 import *
from .rewards import default_winterwheat_reward
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
        self.reward_var = kwargs.get('reward_var', "TWSO")
        # self.reward_var = kwargs.get('reward_var', "NuptakeTotal")
        self.reward_function = reward

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

        self.feature_cost = []
        self.feature_ind = []
        self.feature_ind_dict = OrderedDict()
        self.get_feature_cost_ind()
        # self._env = MeasureOrNot(self._env)

        super().reset(seed=seed)

    def _get_observation_space(self):
        nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self._timestep
        return gym.spaces.Box(0, np.inf, shape=(nvars,))

    def step(self, action):
        """
        Computes customized reward and populates info
        """

        obs, _, terminated, truncated, info = self._env.step(action)

        if isinstance(action, np.ndarray):
            act, measure = action[0], action[1:]

        output = self._env._model.get_output()
        amount = act * self.action_multiplier

        reward, growth = self._get_reward_func(output, amount)

        obs, cost = self.measure_act(obs, measure)

        measurement_cost = sum(cost)

        reward -= measurement_cost

        if 'reward' not in info.keys(): info['reward'] = {}
        info['reward'][self.date] = reward
        if 'growth' not in info.keys(): info['growth'] = {}
        info['growth'][self.date] = growth

        return obs, reward, terminated, truncated, info

    def _get_reward_func(self, output, amount):
        match self.reward_function:  # Needs python 3.10+
            case 'ANE':
                return ane_reward(output, self._env_baseline, self.zero_nitrogen_env_storage, self._timestep,
                                  self.reward_var, self.costs_nitrogen, amount)
            case 'DEF':
                return default_winterwheat_reward(output, self._env_baseline, self.zero_nitrogen_env_storage,
                                                  self._timestep, self.reward_var, self.costs_nitrogen,
                                                  amount)
            case _:
                return default_winterwheat_reward(output, self._env_baseline, self.zero_nitrogen_env_storage,
                                                  self._timestep, self.reward_var, self.costs_nitrogen,
                                                  amount)

    def overwrite_year(self, year):
        self.years = year
        self._env_baseline._agro_management = pcse_gym.envs.common_env.replace_years(
            self._env_baseline._agro_management, year)
        self._env._agro_management = pcse_gym.envs.common_env.replace_years(self._env._agro_management, year)

    def set_location(self, location):
        self._env_baseline._location = location
        self._env_baseline._weather_data_provider = pcse_gym.envs.common_env.get_weather_data_provider(location)
        self._env._location = location
        self._env._weather_data_provider = pcse_gym.envs.common_env.get_weather_data_provider(location)

    def reset(self, seed=None, options=()):
        if isinstance(self.years, list):
            year = self.np_random.choice(self.years)
            self._env_baseline._agro_management = pcse_gym.envs.common_env.replace_years(
                self._env_baseline._agro_management, year)
            self._env._agro_management = pcse_gym.envs.common_env.replace_years(self._env._agro_management, year)

        if isinstance(self.locations, list):
            location = self.locations[self.np_random.choice(len(self.locations), 1)[0]]
            self.set_location(location)

        self._env_baseline.reset(seed=seed)
        obs = self._env.reset(seed=seed)

        # TODO: check whether info should/could be filled
        info = {}

        return obs, info

    def get_feature_cost_ind(self):
        for feature in self.po_features:
            if feature in self.crop_features:
                self.feature_ind.append(self.crop_features.index(feature))
        self.feature_ind = tuple(self.feature_ind)

        for feature in self.crop_features:
            if feature in self.po_features:
                if feature not in self.feature_ind_dict.keys():
                    self.feature_ind_dict[feature] = self.crop_features.index(feature)
        # self.feature_ind = [x + 1 for x in self.feature_ind]

    '''PROTOTYPE
        iterate through feature index from sb3 observation.
        if a measurement action is 0, observation is 0
        otherwise, add cost to getting the measurement'''
    def measure_act(self, obs, measurement):
        costs = self.get_observation_cost()
        measuring_cost = np.zeros(len(measurement))
        for i, i_obs in enumerate(self.feature_ind):
            if not measurement[i]:
                obs[i_obs] = 0
            else:
                measuring_cost[i] = costs[i]
        # TODO: implement a better check for length of measurements
        assert len(measurement) == len(measuring_cost), "Action space and partially observable features are not the" \
                                                        "same length"

        return obs, measuring_cost

    def get_observation_cost(self):
        if not self.feature_cost:
            table = self.list_of_costs()
            # chosen = np.intersect1d(self.po_features, list(table.keys()), assume_unique=True)
            for observed_feature in self.po_features:
                cost = table[observed_feature]
                self.feature_cost.append(cost)
            return self.feature_cost
        else:
            return self.feature_cost
            # TODO: if a variable is not in list_of_costs, define default value
            # if self.po_features not in list(table.keys()):
            #     diff = np.setdiff1d(self.po_features, list(self.list_of_costs().keys()), assume_unique=True)
            #     for val in diff:
            #         self.feature_cost[val] = 1

    @staticmethod
    def list_of_costs():
        lookup = dict(
            LAI=5,
            TAGP=20,
            NAVAIL=20,
            NuptakeTotal=30,
            SM=10
        )
        return lookup

    def valid_action_mask(self):
        # TODO: does this work
        action_masks = np.zeros((self.action_space.n,), dtype=int)

        if self.counter > 3:
            action_masks[0] = 1

        return action_masks

    def render(self, mode="human"):
        pass

    @property
    def date(self) -> datetime.date:
        return self._env._model.day

    @property
    def loc(self) -> datetime.date:
        return self._env._location

    @property
    def features(self):
        return [self.crop_features, self.po_features]

    @property
    def index_feature(self):
        return self._env.index_feature

