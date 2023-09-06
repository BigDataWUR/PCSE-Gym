import gymnasium
import numpy as np
from collections import OrderedDict
from gymnasium import ActionWrapper
import pcse
from datetime import timedelta
import pcse_gym.envs.sb3
from stable_baselines3.common.vec_env import VecNormalize


class ActionConstrainer(ActionWrapper):
    """
    Action Wrapper to limit fertilization actions
    """
    def __init__(self, env, action_limit=0, n_budget=0):
        super(ActionConstrainer, self).__init__(env)
        self.counter = 0
        self.action_limit = action_limit
        self.n_counter = 0
        self.n_budget = n_budget

    def action(self, action):
        if self.action_limit > 0:
            if isinstance(self.action_space, gymnasium.spaces.Discrete):
                action = self.freq_limiter_discrete(action)
            elif isinstance(self.action_space, gymnasium.spaces.MultiDiscrete):
                action = self.freq_limiter_multi_discrete(action)
        if self.n_budget > 0:
            if isinstance(self.action_space, gymnasium.spaces.Discrete):
                action = self.discrete_n_budget(action)
            elif isinstance(self.action_space, gymnasium.spaces.MultiDiscrete):
                action = self.multi_discrete_n_budget(action)
        return action

    def discrete_n_budget(self, action):
        if self.n_counter == self.n_budget:
            action = 0
            return action
        self.n_counter += action * 10
        if self.n_counter > self.n_budget:
            action = ((self.n_budget + action * 10) - self.n_counter) / 10
            self.n_counter = self.n_budget
        return action

    def multi_discrete_n_budget(self, action):
        if self.n_counter == self.n_budget:
            action[0] = 0
            return action
        self.n_counter += action[0] * 10
        if self.n_counter > self.n_budget:
            action[0] = ((self.n_budget + action[0] * 10) - self.n_counter) / 10
            self.n_counter = self.n_budget
        return action

    def freq_limiter_discrete(self, action):
        if action != 0:  # if there's an action, increase the counter
            self.counter += 1
        if self.counter > self.action_limit:  # return 0 if the action exceeds limit
            action = 0
        return action

    def freq_limiter_multi_discrete(self, action):
        if action[0] != 0:
            self.counter += 1
        if self.counter > self.action_limit:
            action[0] = 0
        return action

    def reset(self, **kwargs):
        self.counter = 0
        self.n_counter = 0
        return self.env.reset(**kwargs)


def ratio_rescale(value, old_max=None, old_min=None, new_max=None, new_min=None):
    new_value = (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_value


def non_linear_ratio_rescale(value, old_max=None, old_min=None, new_max=None, new_min=None):
    # Normalize the input value to [0, 1]
    normalized_value = (value - old_min) / (old_max - old_min)

    # Determine which segment we are in based on the normalized value
    if normalized_value <= 0.5:
        # Scale sine to [0, 1] and then rescale to [0.3, 0.55]
        return 0.3 + 0.25 * np.sin(np.pi * normalized_value)
    else:
        # Scale sine to [0, 1] and then rescale to [0.55, 0.8]
        return 0.55 + 0.25 * np.sin(np.pi * (normalized_value - 0.5))


class MeasureOrNot:
    """
    Container to store indexes and index matching logic of the environment observation and the measurement actions
    """
    def __init__(self, env):
        self.env = env
        self.feature_ind = []
        self.feature_ind_dict = OrderedDict()
        self.get_feature_cost_ind()

    # override _observation method, a bit of duplicate code
    def _observation(self, observation, flag=False):
        obs = np.zeros(self.env.observation_space.shape)

        if isinstance(observation, tuple):
            observation = observation[0]

        # start measure logic
        index_feature = OrderedDict()

        for i, feature in enumerate(self.env.crop_features):
            if feature == 'random':
                obs[i] = np.random.default_rng().uniform(0, 10000)
            else:
                obs[i] = observation['crop_model'][feature][-1]

            if feature not in index_feature and not flag and feature in self.env.po_features:
                index_feature[feature] = i
                if len(index_feature.keys()) == len(self.env.po_features):
                    self.index_feature = index_feature

        for i, feature in enumerate(self.env.action_features):
            j = len(self.env.crop_features) + i
            obs[j] = observation['actions'][feature]
        for d in range(self.env.timestep):
            for i, feature in enumerate(self.env.weather_features):
                j = d * len(self.env.weather_features) + len(self.env.crop_features) + len(self.env.action_features) + i
                obs[j] = observation['weather'][feature][d]
        return obs

    # override reset method
    def reset(self, seed=None, return_info=False, options=None):
        obs = super().reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs['actions'] = {'cumulative_nitrogen': 0.0}
        obs['actions'] = {'cumulative_measurement': 0.0}
        return self._observation(obs, flag=True)

    def get_feature_cost_ind(self):
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
        if a measurement action is 0, observation is 0
        if measure action 2, add a noisy observation
        if measure action 1, give correct observation
        """
        measuring_cost = np.zeros(len(measurement))

        for i, i_obs in enumerate(self.feature_ind):
            if not measurement[i]:
                obs[i_obs] = 0.0
            elif measurement[i] == 1:
                measuring_cost[i] = self.get_observation_cost(measurement[i], i_obs)
            else:
                obs[i_obs] = self.get_noise(obs[i_obs], i_obs)
                measuring_cost[i] = self.get_observation_cost(measurement[i], i_obs)
        return obs, measuring_cost

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
                return self.exp_costs()
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
        Potentially enforcing the agent not to dump everything at the start
        Not to be used with CERES 'start-dump'
        Adapted, based on the findings of Raun, W.R. and Johnson, G.V. (1999)
        """
        date_now = self.date - self.start_date
        date_end = self.end_date - self.start_date  # TODO end on flowering?
        recovery = ratio_rescale(date_now / timedelta(days=1),
                                 old_max=date_end / timedelta(days=1), old_min=0.0, new_max=0.8, new_min=0.3)
        return recovery


class VecNormalizePO(VecNormalize):
    """
    A wrapper of the normalizing wrapper for an SB3 vectorized environment
    Overrides the normalization of partially observable crop variables in CropGym
    """
    def __init__(self, venv, *args, **kwargs):
        super(VecNormalizePO, self).__init__(venv, *args, **kwargs)

    def _normalize_obs(self, obs, obs_rms):
        if self.venv.envs[0].unwrapped.sb3_env.index_feature:
            index_feature = self.venv.envs[0].unwrapped.sb3_env.index_feature
            actions = self.venv.unwrapped.actions
            norm = super()._normalize_obs(obs, obs_rms)
            for ind, act in zip(index_feature.values(), actions[1:]):
                if act == 0:
                    norm[ind] = -self.clip_obs  # if no measure, assign lower limit of normalization std
            return norm
        else:
            return super()._normalize_obs(obs, obs_rms)

    def _unnormalize_obs(self, obs, obs_rms):
        if self.venv.envs[0].unwrapped.sb3_env.index_feature:
            index_feature = self.venv.envs[0].unwrapped.sb3_env.index_feature
            actions = self.venv.unwrapped.actions
            unnorm = super()._unnormalize_obs(obs, obs_rms)
            for ind, act in zip(index_feature.values(), actions[1:]):
                if act == 0:
                    unnorm[ind] = 0.0
            return unnorm
        else:
            return super()._unnormalize_obs(obs, obs_rms)

