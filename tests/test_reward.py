import unittest
import numpy as np

import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_reward_dep()
        self.env1 = init_env.initialize_env_reward_ane()

    def test_dep_reward_action(self):
        self.env.reset()
        action = np.array([4])
        _, reward, _, _, _ = self.env.step(action)
        expected_reward = -40 - 10

        self.assertEqual(expected_reward, reward)

    def test_dep_reward_no_action(self):
        self.env.reset()
        action = np.array([0])
        _, reward, _, _, _ = self.env.step(action)
        _, reward, _, _, _ = self.env.step(action)
        _, reward, _, _, _ = self.env.step(action)
        _, reward, _, _, _ = self.env.step(action)
        expected_reward = 0

        self.assertEqual(expected_reward, reward)

    def test_ane_reward(self):
        self.env1.reset()

        for i in range(14):
            action = np.array([4])
            _, reward, _, _, _ = self.env1.step(action)

        expected_reward = ((self.env1.ane.cum_growth - self.env1.ane.cum_baseline_growth)
                           / self.env1.ane.cum_amount) - 4

        self.assertEqual(expected_reward, reward)






