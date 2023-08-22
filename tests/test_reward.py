import unittest
import numpy as np

import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_reward_dep()

    def test_dep_reward(self):
        self.env.reset()
        action = np.array([4])
        _, reward, _, _, _ = self.env.step(action)

        recovery_rate = self.env.sb3_env.recovery_penalty()

        amount = action[0] * self.env.action_multiplier

        recovered_n = amount * recovery_rate
        unrecovered_n = (amount - recovered_n) * 2

        expected_reward = -4 - unrecovered_n - 50

        self.assertEqual(expected_reward, reward)




