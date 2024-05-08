import unittest
import numpy as np

import calendar
import datetime

from pcse_gym.envs.rewards import get_deposition_amount, get_disaggregated_deposition
import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_reward_dep()
        self.env1 = init_env.initialize_env_reward_ane()
        self.end = init_env.initialize_env_end_reward()
        self.eny = init_env.initialize_env_eny_reward()
        self.nue = init_env.initialize_env_nue_reward()
        self.nup = init_env.initialize_env_nup_reward()
        self.har = init_env.initialize_env_har_reward()
        self.dnu = init_env.initialize_env_dnu_reward()
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

    def test_end_reward(self):
        self.test_frequent_action_end_reward(self.end, 2151.55)

    def test_def_reward(self):
        self.test_reward_sp(self.def1, 1823.37)

    def test_end_reward_proper(self):
        self.test_sp_action_end_reward(self.end, 1838.60)

    def test_eny_reward(self):
        self.test_frequent_action_end_reward(self.eny, 8506.63)

    def test_eny_reward_proper(self):
        self.test_sp_action_end_reward(self.eny, 8206.62)

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

    def test_nue_reward(self):
        self.test_frequent_action_end_reward(self.nue, 0)

    def test_nue_reward_proper(self):
        self.test_sp_action_end_reward(self.nue, 1569.55)

    def test_nup_reward(self):
        self.test_reward_sp(self.nup, 142.11)

    def test_har_reward(self):
        self.test_reward_sp(self.har, 1864.22)

    def test_dnu_reward(self):
        self.test_reward_sp(self.dnu, 30.9)

    def test_fin_reward_multiple_years(self):
        self.test_reward_single_year(env=self.fin, year=2002, expected_reward=1405.52)
        self.test_reward_single_year(env=self.fin, year=2005, expected_reward=1337.32)
        self.test_reward_single_year(env=self.fin, year=2020, expected_reward=1105.27)
        self.test_reward_single_year(env=self.fin, year=1990, expected_reward=1434.55)


class NitrogenUseEfficiency(unittest.TestCase):
    def setUp(self):
        self.nue1 = init_env.initialize_env_nue_reward()
        self.def1 = init_env.initialize_env_reward_dep()

    def process_nue(self, n_input, info):
        n_in = self.process_nue_in(n_input)

        return info['NamountSO'][max(info['NamountSO'].keys())] / n_in

    @staticmethod
    def get_days_in_year(year):
        return 365 + calendar.isleap(year)

    def process_nue_in(self, n_input):
        nh4 = 12.5
        no3 = 12.5

        # nh4 = 697 - 0.339 * y
        # no3 = 538.868 - 0.264 * y

        # # date_range1 = (datetime.date(year=y, month=12, day=31) - start).days
        # date_range = (end - start).days
        #
        # nh4_daily = nh4 / self.get_days_in_year(y)
        # no3_daily = no3 / self.get_days_in_year(y)
        #
        # nh4_dis = nh4_daily * date_range
        # no3_dis = no3_daily * date_range

        return n_input + nh4 + no3 + 3.5

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

        calculated_nue = self.process_nue(n_input, info)

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

        calculated_nue = self.process_nue(n_input, info)

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

        calculated_surplus = (self.process_nue_in(n_input)
                              - info['NamountSO'][max(info['NamountSO'].keys())])

        self.assertAlmostEqual(info['Nsurplus'][max(info['Nsurplus'].keys())], calculated_surplus[0], 1)

    def test_disaggregate(self):
        start = datetime.date(2002, 1, 1)
        end = datetime.date(2002, 2, 1)

        nh4_depo, no3_depo = get_deposition_amount(2002)

        daily_nh4 = nh4_depo / 365
        daily_no3 = no3_depo / 365

        expected_nh4 = daily_nh4 * 31
        expected_no3 = daily_no3 * 31

        nh4_dis, no3_dis = get_disaggregated_deposition(year=2002, start_date=start, end_date=end)

        self.assertEqual(expected_nh4, nh4_dis)
        self.assertEqual(expected_no3, no3_dis)

    def test_nue_calc_in_other_rf(self, reward_func='DEP'):
        self.env_rew = init_env.initialize_env(reward=reward_func)
        self.env_rew.overwrite_year(2002)
        self.env_rew.reset()
        terminated = False

        while not terminated:
            _, _, terminated, _, infos = self.env_rew.step(np.array([1]))

        self.assertAlmostEqual(0.58, max(list(infos['NUE'].values())), 0)

    def test_nue_all_rf(self):
        self.test_nue_calc_in_other_rf('DEF')
        self.test_nue_calc_in_other_rf('HAR')
        self.test_nue_calc_in_other_rf('NUE')
        self.test_nue_calc_in_other_rf('FIN')
        self.test_nue_calc_in_other_rf('NUP')
        self.test_nue_calc_in_other_rf('DNU')
        self.test_nue_calc_in_other_rf('END')
        self.test_nue_calc_in_other_rf('GRO')
