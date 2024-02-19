import unittest
import numpy as np

import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_reward_dep()
        self.env1 = init_env.initialize_env_reward_ane()
        self.end = init_env.initialize_env_end_reward()
        self.eny = init_env.initialize_env_eny_reward()

    # def test_dep_reward_action(self):
    #     self.env.reset()
    #     action = np.array([4])
    #     _, reward, _, _, _ = self.env.step(action)
    #     expected_reward = -40 - 10
    #
    #     self.assertEqual(expected_reward, reward)
    #
    # def test_dep_reward_no_action(self):
    #     self.env.reset()
    #     action = np.array([0])
    #     _, reward, _, _, _ = self.env.step(action)
    #     _, reward, _, _, _ = self.env.step(action)
    #     _, reward, _, _, _ = self.env.step(action)
    #     _, reward, _, _, _ = self.env.step(action)
    #     expected_reward = 0
    #
    #     self.assertEqual(expected_reward, reward)

    def test_end_reward(self):
        self.end.reset()

        terminated = False

        while not terminated:
            action = np.array([1])
            _, reward, terminated, _, _ = self.end.step(action)

            if terminated:
                expected_reward = 206.23
            else:
                expected_reward = -10

            self.assertAlmostEqual(expected_reward, reward, 1)

    def test_end_reward_proper(self):
        self.end.reset()

        terminated = False

        week = 0
        n = 12
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = self.end.step(action)

            if terminated:
                expected_reward = 194.41
            elif week == n or week == n + 4:
                expected_reward = -60
            else:
                expected_reward = 0

            self.assertAlmostEqual(expected_reward, reward, 1)

            week += 1

    def test_eny_reward(self):
        self.eny.reset()

        terminated = False

        while not terminated:
            action = np.array([1])
            _, reward, terminated, _, _ = self.eny.step(action)

            if terminated:
                expected_reward = 841.65
            else:
                expected_reward = -10

            self.assertAlmostEqual(expected_reward, reward, 1)

    def test_eny_reward_proper(self):
        self.eny.reset()

        terminated = False

        week = 0
        n = 12
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = self.eny.step(action)

            if terminated:
                expected_reward = 829.83
            elif week == n or week == n + 4:
                expected_reward = -60
            else:
                expected_reward = 0

            self.assertAlmostEqual(expected_reward, reward, 1)

            week += 1

    def test_ane_reward(self):
        self.env1.reset()

        for i in range(14):
            action = np.array([4])
            _, reward, _, _, _ = self.env1.step(action)

        expected_reward = ((self.env1.ane.cum_growth - self.env1.ane.cum_baseline_growth)
                           / self.env1.ane.cum_amount) - 4

        self.assertEqual(expected_reward, reward)






