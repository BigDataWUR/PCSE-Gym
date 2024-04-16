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
                expected_reward = 2151.55
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
                expected_reward = 1943.47
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
                expected_reward = 2747.615
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
                expected_reward = 1115.126
            elif week == n or week == n + 4:
                expected_reward = -60
            else:
                expected_reward = 0

            self.assertAlmostEqual(expected_reward, reward, 1)

            week += 1

    def test_nup_reward(self):
        self.nup.reset()
        terminated = False

        week = 0
        n = 12
        rewards = 0
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = self.nup.step(action)
            rewards += reward
            week += 1

        expected_reward = 142.13

        self.assertAlmostEqual(expected_reward, rewards, 1)

    def test_har_reward(self):
        self.har.reset()
        terminated = False

        week = 0
        n = 12
        rewards = 0
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = self.har.step(action)
            rewards += reward
            week += 1
        expected_reward = 8549.11

        self.assertAlmostEqual(expected_reward, rewards, 1)

    def test_dnu_reward(self):
        self.dnu.reset()
        terminated = False

        week = 0
        n = 12
        rewards = 0
        while not terminated:

            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = self.dnu.step(action)
            rewards += reward
            week += 1

        expected_reward = 25.31

        self.assertAlmostEqual(expected_reward, rewards, 1)

    def test_fin_reward_single_year(self, year=2002, expected_reward=911.42):
        self.fin.overwrite_year(year)
        self.fin.reset()
        terminated = False

        week = 0
        n = 4
        rewards = 0
        while not terminated:

            if week == n or week == n + 4 or week == n + 8:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = self.fin.step(action)
            rewards += reward
            week += 1

        self.assertAlmostEqual(expected_reward, rewards, 1)

    def test_fin_reward_multiple_years(self):
        self.test_fin_reward_single_year(year=2005, expected_reward=807.52)
        self.test_fin_reward_single_year(year=2020, expected_reward=1281.57)
        self.test_fin_reward_single_year(year=1990, expected_reward=1550.18)


class NitrogenUseEfficiency(unittest.TestCase):
    def setUp(self):
        self.nue1 = init_env.initialize_env_nue_reward()
        self.def1 = init_env.initialize_env_reward_dep()

    def process_nue(self, n_input, info, y, start, end):
        n_in = self.process_nue_in(n_input, y, start, end)

        return info['NamountSO'][max(info['NamountSO'].keys())] / n_in

    @staticmethod
    def get_days_in_year(year):
        return 365 + calendar.isleap(year)

    def process_nue_in(self, n_input, y, start, end):
        nh4 = 697 - 0.339 * y
        no3 = 538.868 - 0.264 * y

        # date_range1 = (datetime.date(year=y, month=12, day=31) - start).days
        date_range = (end - start).days

        nh4_daily = nh4 / self.get_days_in_year(y)
        no3_daily = no3 / self.get_days_in_year(y)

        nh4_dis = nh4_daily * date_range
        no3_dis = no3_daily * date_range

        return n_input + nh4_dis + no3_dis + 3.5

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

        calculated_nue = self.process_nue(n_input, info, year, start=self.nue1.sb3_env.agmt.get_start_date(),
                                          end=self.nue1.sb3_env.agmt.get_end_date())

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

        calculated_nue = self.process_nue(n_input, info, year, start=self.nue1.sb3_env.agmt.get_start_date(),
                                          end=self.nue1.sb3_env.agmt.get_end_date())

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

        calculated_surplus = (self.process_nue_in(n_input, year,
                                                  start=self.nue1.sb3_env.agmt.get_start_date(),
                                                  end=self.nue1.sb3_env.agmt.get_end_date())
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

    def test_nue_calc_in_other_rf(self):
        self.def1.overwrite_year(2002)
        self.def1.reset()
        terminated = False

        while not terminated:
            _, _, terminated, _, infos = self.def1.step(np.array([1]))

        self.assertAlmostEqual(0.58, max(list(infos['NUE'].values())), 0)




