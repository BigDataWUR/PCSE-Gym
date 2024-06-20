import unittest
import numpy as np
import datetime

import tests.initialize_env as init_env

from pcse.input.nasapower import NASAPowerWeatherDataProvider
from pcse_gym.utils.nitrogen_utils import (calculate_year_n_deposition,
                                           convert_year_to_n_concentration,
                                           calculate_day_n_deposition)
from pcse_gym.utils.weather_utils.weather_functions import generate_date_list


class TestNitrogenUtils(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env(reward='NUE', pcse_env=2, start_type='sowing')

    def test_n_model_deposition(self):
        year = 2000
        self.env.overwrite_year(year)
        self.env.reset()

        wdp = NASAPowerWeatherDataProvider(52.0, 5.5)
        terminated = False

        while not terminated:
            _, _, terminated, _, _ = self.env.step(np.array([0]))

        date_range = generate_date_list(self.env.sb3_env.agmt.crop_start_date, self.env.sb3_env.agmt.crop_end_date)
        total_rain = 0.0
        daily_rain = [wdp(day).RAIN * 10 for day in date_range]
        for rain in daily_rain:
            total_rain += rain
        print(f"total RAIN {total_rain}")

        nh4test, no3test = calculate_year_n_deposition(year, (52.0, 5.5), self.env.sb3_env.agmt, self.env.sb3_env._site_params)

        self.assertAlmostEqual(nh4test, 14.913303197817111, 0)
        self.assertAlmostEqual(no3test, 25.828005043906614, 0)

    def test_n_concentration_conversion(self):
        nh4, no3 = convert_year_to_n_concentration(2000)

        self.assertEqual(nh4, 2.054742670516606)
        self.assertEqual(no3, 1.1753128075355042)

    def test_day_n_deposition(self):
        year = 2000
        self.env.overwrite_year(year)
        self.env.reset()

        _, _, _, _, info = self.env.step(np.array([0]))
        _, _, _, _, info = self.env.step(np.array([0]))

        rain = list(info['RAIN'].values())[-1]  # mm

        day_nh4_depo, day_no3_depo = calculate_day_n_deposition(rain, self.env.sb3_env._site_params,)

        print(f"day_nh4_depo {day_nh4_depo}")
        print(f"day_no3_depo {day_no3_depo}")

        self.assertAlmostEqual(day_nh4_depo, 0.0014, 1)
        self.assertAlmostEqual(day_no3_depo, 0.0025, 1)
