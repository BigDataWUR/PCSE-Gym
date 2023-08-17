import numpy as np
from collections import OrderedDict
import gymnasium as gym
from gymnasium import ActionWrapper
import pcse
from datetime import timedelta
import pcse_gym.envs.sb3


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


def ratio_rescale(value, old_max=None, old_min=None, new_max=None, new_min=None):
    new_value = (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_value


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
        # for feature in set(self.feature_ind_dict) & set(self.list_of_costs()):

        assert len(measurement) == len(costs), "Action space and partially observable features are not the" \
                                               "same length"
        for i, i_obs in enumerate(self.feature_ind):
            if not measurement[i]:
                obs[i_obs] = 0  # might want to change this
            else:
                measuring_cost[i] = costs[i]
        return obs, measuring_cost

    def get_observation_cost(self):
        if not self.feature_cost:
            for observed_feature in self.env.po_features:
                if observed_feature not in self.list_of_costs():
                    self.feature_cost.append(1)
                else:
                    self.feature_cost.append(self.list_of_costs()[observed_feature])
            return self.feature_cost
        else:
            return self.feature_cost

    @staticmethod
    def list_of_costs():
        lookup = dict(
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
        return lookup


class VariableRecoveryRate(pcse_gym.envs.sb3.StableBaselinesWrapper):
    def __init__(self, env):
        super().__init__()
        self.__dict__.update(env.__dict__)

    def _apply_action(self, action):
        amount = action * self.action_multiplier
        recovery_rate = self.recovery_penalty()
        self._model._send_signal(signal=pcse.signals.apply_n, N_amount=amount * 10, N_recovery=recovery_rate,
                                 amount=amount, recovery=recovery_rate)

    def recovery_penalty(self):
        """
        estimation function due to static recovery rate of WOFOST/LINTUL
        Potentially enforcing the agent not to dump everything at the start (to be tested)
        Not to be used with CERES 'start-dump'
        Adapted, based on the findings of Raun, W.R. and Johnson, G.V. (1999)
        """
        date_now = self.date - self.start_date
        date_end = self.end_date - self.start_date  # TODO end on flowering?
        recovery = ratio_rescale(date_now / timedelta(days=1),
                                 old_max=date_end / timedelta(days=1), old_min=0.0, new_max=0.8, new_min=0.3)
        return recovery


