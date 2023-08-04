import unittest
import os
import gymnasium as gym

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from pcse_gym.utils.eval import evaluate_policy, summarize_results
import initialize_env as init_env


# TODO: figure out a more robust way of obtaining the file paths
class TestModel(unittest.TestCase):
    def test_model(self):
        model_path = os.path.abspath('tests/model-1')
        stats_path = os.path.abspath('tests/model-1.pkl')
        custom_objects = {"lr_schedule": lambda x: 0.0002, "clip_range": lambda x: 0.3}
        custom_objects["action_space"] = gym.spaces.Discrete(3)
        model_cropgym = PPO.load(model_path, custom_objects=custom_objects, device='cuda', print_system_info=True)
        rewards_model, results_model = {}, {}
        test_years = [1992, 2002]
        test_locations = [(52, 5.5), (48, 0)]
        for test_year in test_years:
            for location in list(set(test_locations)):
                env = init_env.initialize_env(pcse_env=0, crop_features=init_env.get_crop_features(pcse_env=0), nitrogen_levels=3,
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