import unittest
import numpy as np

import calendar
import datetime

import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.fin = init_env.initialize_env_fin_reward()
        self.def1 = init_env.initialize_env(reward='DEF')

    def test_reward_sp(self, env=None, expected_reward=None):
        env.reset() if env is not None else env
        terminated = False

        week = 0
        n = 12
        rewards = 0
        while not terminated:
            if env is None:
                break
            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = env.step(action)
            rewards += reward
            week += 1
        if expected_reward:
            self.assertAlmostEqual(expected_reward, rewards, 0)

    def test_frequent_action_end_reward(self, env=None, expected_reward=None):
        env.reset() if env is not None else env

        terminated = False
        expected = expected_reward

        while not terminated:
            if env is None:
                break
            action = np.array([1])
            _, reward, terminated, _, _ = env.step(action)

            if terminated:
                expected = expected_reward
            elif not terminated and env.reward_function != 'NUE':
                expected = -10

            if expected_reward:
                self.assertAlmostEqual(expected, reward, 0)

    def test_sp_action_end_reward(self, env=None, expected_reward=None, year=None):
        if year is not None:
            env.overwrite_year(year)
            env.reset()

        terminated = False

        week = 0
        n = 12
        while not terminated:
            if env is None:
                break
            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = env.step(action)

            if terminated:
                expected = expected_reward
            elif (week == n or week == n + 4) and env.reward_function != 'NUE':
                expected = -60
            else:
                expected = 0

            self.assertAlmostEqual(expected, reward, 0)

            week += 1

    def test_reward_single_year(self, env=None, year=2002, expected_reward=None):
        if env is not None:
            env.overwrite_year(year)
            env.reset()
        terminated = False

        week = 0
        n = 4
        rewards = 0
        while not terminated:
            if env is None:
                break
            if week == n or week == n + 4 or week == n + 8:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = env.step(action)
            rewards += reward
            week += 1

        if expected_reward:
            self.assertAlmostEqual(expected_reward, rewards, 0)

    def test_def_reward(self):
        self.test_reward_sp(self.def1, 1823.37)

    def test_fin_reward_multiple_years(self):
        self.test_reward_single_year(env=self.fin, year=2002, expected_reward=1405.52)
        self.test_reward_single_year(env=self.fin, year=2005, expected_reward=1337.32)
        self.test_reward_single_year(env=self.fin, year=2020, expected_reward=1105.27)
        self.test_reward_single_year(env=self.fin, year=1990, expected_reward=1434.55)
