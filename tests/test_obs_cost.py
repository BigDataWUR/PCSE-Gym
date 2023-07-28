import unittest
import numpy as np
import time
import warnings
import os
import lib_programname
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_model_kwargs
from pcse_gym.utils.defaults import *
from pcse_gym.utils.eval import FindOptimum, evaluate_policy, summarize_results

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_rootdir():
    path_to_program = lib_programname.get_path_executed_script()
    rootdir = path_to_program.parents[1]
    return rootdir


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
                   costs_nitrogen=10, reward='DEF', nitrogen_levels=7, action_multiplier=1.0, add_random=False,
                   years=get_default_train_years(), locations=get_default_location()):
    if add_random:
        po_features.append('random'), crop_features.append('random')
    action_space = get_action_space(nitrogen_levels=nitrogen_levels, po_features=po_features)
    kwargs = dict(po_features=po_features, args_measure=po_features is not None)
    env_return = WinterWheat(crop_features=crop_features,
                             costs_nitrogen=costs_nitrogen,
                             years=years,
                             locations=locations,
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


class TestModel(unittest.TestCase):
    def test_model(self):
        rootdir = get_rootdir()
        model_path = os.path.join(rootdir, 'tests/model-1.zip')
        stats_path = os.path.join(rootdir, 'tests/model-1.pkl')
        custom_objects = {"lr_schedule": lambda x: 0.0002, "clip_range": lambda x: 0.3}
        custom_objects["action_space"] = gym.spaces.Discrete(3)
        model_cropgym = PPO.load(model_path, custom_objects=custom_objects, device='cuda', print_system_info=True)
        rewards_model, results_model = {}, {}
        test_years = [1992, 2002]
        test_locations = [(52, 5.5), (48, 0)]
        for test_year in test_years:
            for location in list(set(test_locations)):
                env = initialize_env(pcse_env=0, crop_features=get_crop_features(pcse_env=0), nitrogen_levels=3,
                                     action_multiplier=2.0, years=test_year, locations=location)
                env = DummyVecEnv([lambda: env])
                env = VecNormalize.load(stats_path, env)
                env.training, env.norm_reward = False, True
                results_key = (test_year, location)
                rewards_model[results_key], results_model[results_key] = evaluate_policy(model_cropgym, env, amount=1)
        summary = summarize_results(results_model)
        self.assertAlmostEqual(summary.loc[[(1992, (48, 0))]]['WSO'].values[0], 357.7, 1)
        self.assertAlmostEqual(summary.loc[[(1992, (48, 0))]]['reward'].values[0], -74.8, 1)
        self.assertAlmostEqual(summary.loc[[(2002, (52, 5.5))]]['WSO'].values[0], 880.5, 1)
        self.assertAlmostEqual(summary.loc[[(2002, (52, 5.5))]]['reward'].values[0], 133.0, 1)


if __name__ == '__main__':
    unittest.main()
