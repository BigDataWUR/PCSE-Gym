from collections import OrderedDict, defaultdict
from copy import deepcopy

import numpy as np


class MeasureOrNot:
    """
    Container to store indexes and index matching logic of the environment observation and the measurement actions
    """
    def __init__(self, env, extend_obs=False, placeholder_val=-1.11):
        self.env = env
        self.feature_ind = []
        self.feature_ind_dict = OrderedDict()
        self.measure_freq = defaultdict(dict)
        self.get_feature_cost_ind()
        self.mask = extend_obs
        self.placeholder = placeholder_val

    def get_feature_cost_ind(self) -> None:
        for feature in self.env.po_features:
            if feature in self.env.crop_features:
                self.feature_ind.append(self.env.crop_features.index(feature))
        self.feature_ind = tuple(self.feature_ind)

        for feature in self.env.crop_features:
            if feature in self.env.po_features and feature not in self.feature_ind_dict:
                self.feature_ind_dict[feature] = self.env.crop_features.index(feature)

    def measure_act(self, obs, measurement):
        """
        iterate through feature index from sb3 observation.
        if a measurement action is 0, observation is ::self.placeholder
        if measure action 2, add a noisy observation
        if measure action 1, give correct observation
        """
        measuring_cost = np.zeros(len(measurement))

        for i, i_obs in enumerate(self.feature_ind):
            if not measurement[i]:
                obs[i_obs] = self.placeholder
            elif measurement[i] == 1:
                measuring_cost[i] = self.get_observation_cost(measurement[i], i_obs)
            else:
                obs[i_obs] = self.get_noise(obs[i_obs], i_obs)
                measuring_cost[i] = self.get_observation_cost(measurement[i], i_obs)
            self.set_measure_freq(measurement)

        if self.mask:
            obs = self.extend_observation(obs)

        return obs, measuring_cost

    def extend_observation(self, obs):
        obs_ = deepcopy(obs)
        trunc_obs = obs[:len(obs)//2]  # np.append(obs, np.zeros(len(obs)))
        # binary observation vector, explicitly stating if an observation is masked
        for i, ori_ob in enumerate(trunc_obs):
            j = len(obs)//2 + i
            if ori_ob == self.placeholder:
                obs_[j] = 0
            else:
                obs_[j] = 1
        return obs_

    def get_noise(self, obs, index):
        rng = np.random.default_rng()
        feature = self.get_match(index)
        match feature:
            case 'LAI':
                obs = rng.normal(obs, 0.4)
            case 'SM':
                obs = rng.normal(obs, 0.2)
            case 'NAVAIL':
                obs = rng.normal(obs, 5)
            case 'NuptakeTotal':
                obs = rng.normal(obs, 5)
            case 'TAGP':
                obs = rng.normal(obs, 2)
            case _:
                obs = rng.normal(obs, 1)
        return obs

    def get_observation_cost(self, price, ind):
        costs = self.list_of_costs(price)
        key_cost = self.get_match(ind)
        value_cost = costs.get(key_cost, 1)
        return value_cost

    def list_of_costs(self, cost):
        match cost:
            case 1:
                return self.get_cost_function_coef()
            case 2:
                return self.cheap_costs()
            case _:
                return Exception("Not a valid choice")

    def get_match(self, reference):
        key_match = None
        for key, value in self.feature_ind_dict.items():
            if value == reference:
                key_match = key
                break
        return key_match

    def get_measure_cost_vector(self):
        exp_costs = self.exp_costs()
        vector = [exp_costs.get(k) for k in self.feature_ind_dict if k in exp_costs]
        return vector

    @staticmethod
    def exp_costs():
        return dict(
            LAI=1,
            TAGP=5,
            NAVAIL=5,
            NuptakeTotal=3,
            SM=1,
            TGROWTH=5,
            TNSOIL=5,
            NUPTT=3,
            TRAIN=1
        )

    @staticmethod
    def cheap_costs():
        return dict(
            LAI=0.5,
            TAGP=2.5,
            NAVAIL=2.5,
            NuptakeTotal=2,
            SM=0.5,
            TGROWTH=2.5,
            TNSOIL=2.5,
            NUPTT=2,
            TRAIN=0.5
        )

    @staticmethod
    def no_costs():
        return dict(
            LAI=0,
            TAGP=0,
            NAVAIL=0,
            NuptakeTotal=0,
            SM=0,
            TGROWTH=0,
            TNSOIL=0,
            NUPTT=0,
            TRAIN=0,
            random=0,
        )

    @staticmethod
    def same_costs(cost=3):
        return dict(
            LAI=cost,
            TAGP=cost,
            NAVAIL=cost,
            NuptakeTotal=cost,
            SM=cost,
            TGROWTH=cost,
            TNSOIL=cost,
            NUPTT=cost,
            TRAIN=cost,
            random=cost
        )

    def get_cost_function_coef(self):
        if self.env.cost_measure == 'real':
            return self.exp_costs()
        elif self.env.cost_measure == 'no':
            return self.no_costs()
        elif self.env.cost_measure == 'same':
            return self.same_costs()
        else:
            raise Exception(f"Error! {self.env.cost_measure} not a valid choice")

    def set_measure_freq(self, measurement):
        date_key = self.get_date_key()
        loc_key = self.get_loc_key()
        did = {}
        for k, v in self.feature_ind_dict.items():
            for ind, vi in enumerate(self.feature_ind):
                if vi == v:
                    did[k] = measurement[ind]
        self.measure_freq[loc_key][date_key] = did

    def get_year_key(self):
        year = self.env.date.year
        return f'{year}'

    def get_loc_key(self):
        location = self.env.loc
        return f'{location}'

    def get_date_key(self):
        date = self.env.date#.strftime('%m-%d')
        return f'{date}'

    @property
    def measure_len(self):
        return len(self.env.po_features)
