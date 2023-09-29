import gymnasium as gym
import numpy as np
from collections import OrderedDict
from gymnasium import ActionWrapper
import pcse
from datetime import timedelta
import pcse_gym.envs.sb3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd
from copy import deepcopy


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
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.freq_limiter_discrete(action)
            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                action = self.freq_limiter_multi_discrete(action)
        if self.n_budget > 0:
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.discrete_n_budget(action)
            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
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
            action = action.copy()  # Needed for RLlib VecEnvs
            action[0] = 0
            return action
        self.n_counter += action[0] * 10
        if self.n_counter > self.n_budget:
            action = action.copy()
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
            action = action.copy()
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

    def _apply_action(self, action) -> None:
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


def generate_combinations(elements):
    from itertools import combinations
    all_combinations = []

    for x in range(1, len(elements) + 1):
        all_combinations.extend(combinations(elements, x))

    return all_combinations


class VecNormalizePO(VecNormalize):
    """
    A wrapper of the normalizing wrapper for an SB3 vectorized environment
    Overrides the normalization of partially observable crop variables in CropGym
    """
    def __init__(self, venv, *args, **kwargs):
        super(VecNormalizePO, self).__init__(venv, *args, **kwargs)
        self.obs_rms = RunningMeanStdPO(self.observation_space.shape, index=self.venv.envs[0].unwrapped.sb3_env.index_feature)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                if self.venv.envs[0].unwrapped.sb3_env.index_feature and self.venv.envs[0].unwrapped.sb3_env.step_check:
                    self.obs_rms.update_index(self.get_no_measure(), obs)
                else:
                    self.obs_rms.update(obs)

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)
        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)
        return self.normalize_obs(obs)

    def normalize_obs(self, obs):
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        obs_ = deepcopy(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                # Only normalize the specified keys
                for key in self.norm_obs_keys:
                    obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).astype(np.float32)
            else:
                obs_ = self._normalize_obs(obs_, self.obs_rms).astype(np.float32)
        return obs_

    def get_no_measure(self):
        index_feature = self.venv.envs[0].unwrapped.sb3_env.index_feature
        actions = self.venv.unwrapped.actions
        measure_index = [i for i, e in enumerate(actions[0][1:]) if e == 0]
        if len(measure_index) > 0:
            # remove feat. from rms if not measured
            ind_feat = np.delete(list(index_feature.values()), measure_index)
        else:
            ind_feat = []
        return ind_feat


class RunningMeanStdPO(RunningMeanStd):
    def __init__(self, epsilon=1e-4, shape=(), index=[]):
        super(RunningMeanStdPO, self).__init__(epsilon=1e-4, shape=())
        self.obs_shape = shape
        self.index_po_full = index
        self.index_po = index
        self.index_po_dict = {key: 0.0 for key in self.index_po_full}
        self.index_po_mean = None
        self.index_po_var = None
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update_index(self, index, arr):
        self.index_po = index
        self.update(arr)

    def update_index_value(self):
        for key, value in zip(self.index_po, self.mean):
            self.index_po_mean[key] = value
        for key, value in zip(self.index_po, self.var):
            self.index_po_var[key] = value

    def update(self, arr: np.ndarray) -> None:
        if len(self.index_po) > 0:
            arr = np.array([val for i, val in enumerate(arr[0]) if i in self.index_po])
            self.mean = np.array([val for i, val in enumerate(self.mean) if i in self.index_po])
            self.var = np.array([val for i, val in enumerate(self.var) if i in self.index_po])
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

        self.update_index_value()



class ClipNormalizeObservation(gym.wrappers.NormalizeObservation):
    """
    Wrapper to add capability of clipping the observation running std.
    Mimicking SB3 VecNormalize.
    """
    def __init__(self, venv, clip, *args, **kwargs):
        super(ClipNormalizeObservation, self).__init__(venv, *args, **kwargs)
        self.clip = clip

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip, self.clip)


class ClipNormalizeReward(gym.wrappers.NormalizeReward):
    """
    Wrapper to add capability of clipping the observation running std.
    Mimicking SB3 VecNormalize.
    """

    def __init__(self, venv, clip, *args, **kwargs):
        super(ClipNormalizeReward, self).__init__(venv, *args, **kwargs)
        self.clip = clip

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip, self.clip)



