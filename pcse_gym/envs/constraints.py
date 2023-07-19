import numpy as np
from collections import OrderedDict
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper


# TODO, limit fertilization actions; WIP
class ActionLimiter(ActionWrapper):
    def __init__(self, env, action_limit):
        super(ActionLimiter, self).__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.counter = 0
        self.action_limit = action_limit

    def valid_action_mask(self):
        # TODO: does this work
        action_masks = np.zeros((self.action_space.n,), dtype=int)

        if self.counter > 3:
            action_masks[0] = 1

        return action_masks

    def action(self, action):
        if action != 0:  # if there's an action, increase the counter
            self.counter += 1
        if self.counter > self.action_limit:  # return 0 if the action exceeds limit
            action = 0
        return action

    def reset(self, **kwargs):
        self.counter = 0
        return self.env.reset(**kwargs)


class MeasureOrNot:
    """
    Container to store indexes and index matching logic of the environment observation and the measurement actions
    """

    def __init__(self, env):
        self.env = env
        self.feature_cost = []
        self.feature_ind = []
        self.feature_ind_dict = OrderedDict()
        self.get_feature_cost_ind()

    def get_feature_cost_ind(self):
        for feature in self.env.po_features:
            if feature in self.env.crop_features:
                self.feature_ind.append(self.env.crop_features.index(feature))
        self.feature_ind = tuple(self.feature_ind)

        for feature in self.env.crop_features:
            if feature in self.env.po_features:
                if feature not in self.feature_ind_dict.keys():
                    self.feature_ind_dict[feature] = self.env.crop_features.index(feature)

    def measure_act(self, obs, measurement):
        """PROTOTYPE
        iterate through feature index from sb3 observation.
        if a measurement action is 0, observation is 0
        otherwise, add cost to getting the measurement"""
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
            for observed_feature in self.env.po_features:
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
            LAI=1,
            TAGP=5,
            NAVAIL=5,
            NuptakeTotal=3,
            SM=1
        )
        return lookup
