import unittest
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import *
from pcse_gym.utils.defaults import *


def initialize_env():
    po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal']

    po_dicts = dict(po_features=po_features)

    env_pcse_eval = WinterWheat(crop_features=get_wofost_default_crop_features(),
                                action_features=get_default_action_features(),
                                weather_features=get_default_weather_features(),
                                costs_nitrogen=10, years=get_default_train_years(),
                                locations=get_default_location(),
                                action_space=gym.spaces.Discrete(7), action_multiplier=1.0, reward='DEF',
                                **get_pcse_model(1), **po_dicts)
    return env_pcse_eval


class TestCosts(unittest.TestCase):

    def test_oc(self):
        env = initialize_env()

        cost = env.get_observation_cost()

        obs = np.zeros(30)

        measure = [1, 1, 1, 1, 1]

        _, measurement_cost = env.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))


if __name__ == '__main__':
    unittest.main()
