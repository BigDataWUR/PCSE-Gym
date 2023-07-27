import unittest
import numpy as np
import time
import warnings
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_model_kwargs
from pcse_gym.utils.defaults import *
from pcse_gym.utils.eval import FindOptimum

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_po_features(pcse_env=1):
    if pcse_env:
        po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal']
    else:
        po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
    return po_features


def get_crop_features(pcse_env=1):
    if pcse_env:
        crop_features = get_wofost_default_crop_features()
    else:
        crop_features = get_lintul_default_crop_features()
    return crop_features


def get_action_space(nitrogen_levels=7, po_features=[]):
    if po_features:
        a_shape = [nitrogen_levels] + [2] * len(po_features)
        space_return = gym.spaces.MultiDiscrete(a_shape)
    else:
        space_return = gym.spaces.Discrete(nitrogen_levels)
    return space_return


def initialize_env(pcse_env=1, po_features=[], crop_features=get_crop_features(pcse_env=1),
                   costs_nitrogen=10, reward='DEF', action_multiplier=1.0, add_random=False):
    if add_random:
        po_features.append('random'), crop_features.append('random')
    action_space = get_action_space(nitrogen_levels=7, po_features=po_features)
    kwargs = dict(po_features=po_features, args_measure=po_features is not None)
    env_return = WinterWheat(crop_features=crop_features,
                             costs_nitrogen=costs_nitrogen,
                             years=get_default_train_years(),
                             locations=get_default_location(),
                             action_space=action_space,
                             action_multiplier=action_multiplier,
                             reward=reward,
                             **get_model_kwargs(pcse_env),
                             **kwargs)

    return env_return


def initialize_env_po():
    return initialize_env(po_features=get_po_features())


def initialize_env_no_baseline():
    return initialize_env(reward='GRO', action_multiplier=2.0)


def initialize_env_random():
    return initialize_env(po_features=get_po_features(), add_random=True)


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.env = initialize_env_po()

    def test_oc(self):
        self.env.reset()
        cost = self.env.measure_features.get_observation_cost()

        obs = np.ones(30)

        measure = [1, 1, 1, 1, 1]

        _, measurement_cost = self.env.measure_features.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))

    def test_index_match_with_obs(self):
        self.env.reset()
        action = np.array([0, 1, 0, 0, 0, 1])
        # measure = action[1:]
        start = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        end = time.time()
        print(f'time: {end - start}')
        actual = []
        for i in range(1, 4):
            a = obs[self.env.measure_features.feature_ind[i]]
            actual.append(a)
        expected = list(np.zeros(3))

        self.assertListEqual(expected, actual)


class TestNoMeasure(unittest.TestCase):
    def setUp(self):
        self.env = initialize_env_no_baseline()

    def test_output(self):
        self.env.reset()
        action = np.array([3])
        _, _, _, _, _ = self.env.step(action)
        action = np.array([4])
        obs, reward, terminated, truncated, info = self.env.step(action)

        expected = -80
        actual = reward

        self.assertEqual(expected, actual)


class TestRandomFeature(unittest.TestCase):
    def setUp(self):
        self.env = initialize_env_random()

    def test_random(self):
        self.env.reset()
        action = np.array([0, 1, 0, 0, 0, 1, 1])
        measure = action[1:]
        obs, reward, terminated, truncated, info = self.env.step(action)
        expected = []
        costs = self.env.measure_features.get_observation_cost()
        for i, cost in zip(measure, costs):
            if i:
                expected.append(cost)
            else:
                expected.append(0)
        # expected = env3.measure_features.list_of_costs()
        expected = -float(sum(expected))

        self.assertEqual(expected, reward)


class TestCeres(unittest.TestCase):
    def setUp(self):
        self.env = initialize_env(pcse_env=0, crop_features=get_crop_features(pcse_env=0))
        self.env = VecNormalize(DummyVecEnv([lambda: self.env]), norm_reward=True, clip_reward=50., gamma=1)

    def test_single_year(self):
        ceres_result = FindOptimum(self.env, [1992]).optimize_start_dump().item()
        self.assertAlmostEqual(17.6, ceres_result, 1)

    def test_multiple_years(self):
        ceres_result = FindOptimum(self.env, [1992, 2002]).optimize_start_dump().item()
        self.assertAlmostEqual(19.1, ceres_result, 1)


if __name__ == '__main__':
    unittest.main()
