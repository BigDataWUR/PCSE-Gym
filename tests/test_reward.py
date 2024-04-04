import unittest
import numpy as np

import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_reward_dep()
        self.env1 = init_env.initialize_env_reward_ane()
        self.end = init_env.initialize_env_end_reward()
        self.eny = init_env.initialize_env_eny_reward()
        self.nue = init_env.initialize_env_nue_reward()

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

    def test_end_reward(self):
        self.end.reset()

        terminated = False

        while not terminated:
            action = np.array([1])
            _, reward, terminated, _, _ = self.end.step(action)

            if terminated:
                expected_reward = 2151.22
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
                expected_reward = 1943.18
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
                expected_reward = 8506.63
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
                expected_reward = 8298.59
            elif week == n or week == n + 4:
                expected_reward = -60
            else:
                expected_reward = 0

            self.assertAlmostEqual(expected_reward, reward, 1)

            week += 1

    def test_nue_reward(self):
        self.nue.reset()

        terminated = False

        while not terminated:
            action = np.array([1])
            _, reward, terminated, _, _ = self.nue.step(action)

            if terminated:
                expected_reward = 2546.4
            else:
                expected_reward = -10

            self.assertAlmostEqual(expected_reward, reward, 1)

    def test_nue_reward_proper(self):
        self.nue.reset()

        terminated = False

        week = 0
        n = 12
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = self.nue.step(action)

            if terminated:
                expected_reward = 1330.2
            elif week == n or week == n + 4:
                expected_reward = -60
            else:
                expected_reward = 0

            self.assertAlmostEqual(expected_reward, reward, 1)

            week += 1


class NitrogenUseEfficiency(unittest.TestCase):
    def setUp(self):
        self.nue1 = init_env.initialize_env_nue_reward()

    def process_nue(self, n_input, info, y):
        n_in = self.process_nue_in(n_input, info, y)

        return info['NamountSO'][max(info['NamountSO'].keys())] / n_in

    def process_nue_in(self, n_input, info, y):
        nh4 = 697 - 0.339 * y
        no3 = 538.868 - 0.264 * y
        n_depo = nh4 + no3

        return n_input + 3.5 + n_depo

    def test_nue_value(self):
        year = 2002
        self.nue1.overwrite_year(year)
        self.nue1.reset()

        terminated = False
        info = None
        n_input = 0

        while not terminated:
            action = np.array([1])
            _, rew, terminated, _, info = self.nue1.step(action)
            n_input += list(info['fertilizer'].values())[0]

        calculated_nue = self.process_nue(n_input, info, year)

        self.assertAlmostEqual(info['NUE'][max(info['NUE'].keys())], calculated_nue, 1)

    def test_nue_value_proper(self):
        year = 2002
        self.nue1.overwrite_year(year)
        self.nue1.reset()

        terminated = False
        info = None
        n_input = 0

        week = 0
        n = 12
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, info = self.nue1.step(action)
            n_input += list(info['fertilizer'].values())[0]

        calculated_nue = self.process_nue(n_input, info, year)

        self.assertAlmostEqual(info['NUE'][max(info['NUE'].keys())], calculated_nue, 1)

    def test_nue_surplus(self):
        year = 2002
        self.nue1.overwrite_year(year)
        self.nue1.reset()

        terminated = False
        info = None
        n_input = 0

        week = 0
        n = 12
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, info = self.nue1.step(action)
            n_input += action * 10

        calculated_surplus = self.process_nue_in(n_input, info, year) - info['NamountSO'][max(info['NamountSO'].keys())]

        self.assertAlmostEqual(info['Nsurplus'][max(info['Nsurplus'].keys())], calculated_surplus, 1)