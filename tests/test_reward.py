import unittest
import numpy as np
from math import isclose

import calendar
import datetime

from pcse_gym.envs.rewards import get_deposition_amount, get_disaggregated_deposition
import tests.initialize_env as init_env


class Rewards(unittest.TestCase):
    def setUp(self):
        self.dep = init_env.initialize_env_reward_dep()
        self.eny = init_env.initialize_env_eny_reward()
        self.nue = init_env.initialize_env_nue_reward()
        self.nup = init_env.initialize_env_nup_reward()
        self.har = init_env.initialize_env_har_reward()
        self.dnu = init_env.initialize_env_dnu_reward()
        self.fin = init_env.initialize_env_fin_reward()
        self.def1 = init_env.initialize_env(reward='DEF', pcse_env=2)
        self.dne = init_env.initialize_env(reward='DNE', pcse_env=2)

    @staticmethod
    def run_steps_sp(env, year, terminated):
        env.overwrite_year(year)
        env.reset()

        week = 0
        n = 8
        rewards = 0
        while not terminated:
            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = env.step(action)
            rewards += reward
            week += 1

        return rewards

    def test_reward_functions(self):
        rfs = [self.dep, self.nue, self.eny, self.nup,
               self.har, self.dnu, self.fin, self.def1, self.dne]
        expected_rs = [8717.19, 998.8, 8737.19, 145.01, 1278.03, 9.07, 1372.10, 2111.23, 1000.05,]
        for rf_env, expected_r in zip(rfs, expected_rs):
            r = self.run_steps_sp(rf_env, 2002, False)
            check_if_close = isclose(r, expected_r, rel_tol=2)
            self.assertTrue(check_if_close)


class NitrogenUseEfficiency(unittest.TestCase):
    def setUp(self):
        self.nue1 = init_env.initialize_env_nue_reward()
        self.def1 = init_env.initialize_env_reward_dep()

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
        self.env_rew = init_env.initialize_env(reward=reward_func, pcse_env=2)
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
