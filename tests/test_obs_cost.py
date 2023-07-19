import unittest
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import *
from pcse_gym.utils.defaults import *
import time

def initialize_env():




    po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal']

    po_dicts = dict(po_features=po_features)

    a_shape = [7] + [2] * len(po_features)
    action_spaces = gym.spaces.MultiDiscrete(a_shape)

    env_pcse_eval = WinterWheat(crop_features=get_wofost_default_crop_features(),
                                action_features=get_default_action_features(),
                                weather_features=get_default_weather_features(),
                                costs_nitrogen=10, years=get_default_train_years(),
                                locations=get_default_location(),
                                action_space=action_spaces, action_multiplier=1.0, reward='DEF',
                                **get_pcse_model(1), **po_dicts)

    return env_pcse_eval


class TestCosts(unittest.TestCase):

    def test_oc(self):
        env = initialize_env()

        cost = env.measure_features.get_observation_cost()

        obs = np.ones(30)

        measure = [1, 1, 1, 1, 1]

        _, measurement_cost = env.measure_features.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))

    def test_index_match_with_obs(self):
        env = initialize_env()

        action = np.array([0, 1, 0, 0, 0, 1])
        # measure = action[1:]
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        end = time.time()
        print(end - start)
        self.assertEqual(obs[env.measure_features.feature_ind[1]], 0.0)
        self.assertEqual(obs[env.measure_features.feature_ind[2]], 0.0)
        self.assertEqual(obs[env.measure_features.feature_ind[3]], 0.0)


if __name__ == '__main__':
    unittest.main()
