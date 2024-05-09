import unittest
import numpy as np

from initialize_env import initialize_env


class ZeroEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.years = [*range(1990, 2022)]
        self.env = initialize_env(reward="DEF", start_type='sowing')

    def test_zero_env(self):
        """Test whether zero env is referencing the correct year by
        checking if the DVS in the zero env and RL env is the same"""
        terminated = False
        for year in self.years:
            self.env.overwrite_year(year)
            self.env.reset()
            while not terminated:
                _, _, terminated, _, info = self.env.step(np.array([0]))
            dvs_rl = list(info['DVS'].values())
            dvs_zero = list(self.env.zero_nitrogen_env_storage.get_result[f'{year}-(52, 5.5)']['DVS'].values())
            terminated = False
            print(f'year {year}')
            self.assertListEqual(dvs_rl, dvs_zero)  # add assertion here
