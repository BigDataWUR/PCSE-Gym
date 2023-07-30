import unittest
from initialize_env import *
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pcse_gym.utils.eval import FindOptimum


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


