import unittest
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import *
from pcse_gym.utils.defaults import *
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def initialize_env():
    pcse_env = 0
    if pcse_env:
        po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal']
        crop_features = get_wofost_default_crop_features()
    else:
        po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
        crop_features = get_lintul_default_crop_features()

    kwargs = dict(po_features=po_features, args_measure=True)

    a_shape = [7] + [2] * len(po_features)
    action_spaces = gym.spaces.MultiDiscrete(a_shape)

    env_pcse_eval = WinterWheat(crop_features=crop_features,
                                action_features=get_default_action_features(),
                                weather_features=get_default_weather_features(),
                                costs_nitrogen=10, years=get_default_train_years(),
                                locations=get_default_location(), all_years=get_default_years(),
                                all_locations=get_default_location(),
                                action_space=action_spaces, action_multiplier=1.0, reward='DEF',
                                **get_pcse_model(pcse_env), **kwargs)

    return env_pcse_eval


def initialize_no_baseline():

    action_spaces = gym.spaces.Discrete(7)

    env_pcse_eval = WinterWheat(crop_features=get_wofost_default_crop_features(),
                                action_features=get_default_action_features(),
                                weather_features=get_default_weather_features(),
                                costs_nitrogen=10, years=get_default_train_years(),
                                locations=get_default_location(), all_years=get_default_years(),
                                all_locations=get_default_location(),
                                action_space=action_spaces, action_multiplier=2.0, reward='GRO',
                                **get_pcse_model(1))

    return env_pcse_eval


env = initialize_env()

env2 = initialize_no_baseline()


class TestMeasure(unittest.TestCase):

    def test_oc(self):
        env.reset()
        cost = env.measure_features.get_observation_cost()

        obs = np.ones(30)

        measure = [1, 1, 1, 1, 1]

        _, measurement_cost = env.measure_features.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))

    def test_index_match_with_obs(self):
        env.reset()
        action = np.array([0, 1, 0, 0, 0, 1])
        # measure = action[1:]
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        end = time.time()
        print(end - start)
        actual = []
        for i in range(1, 4):
            a = obs[env.measure_features.feature_ind[i]]
            actual.append(a)
        expected = list(np.zeros(3))

        self.assertListEqual(actual, expected)


class TestNoMeasure(unittest.TestCase):

    def test_output(self):
        env2.reset()
        action = np.array([3])
        obs, reward, terminated, truncated, info = env2.step(action)

        expected = -60

        self.assertEqual(reward, expected)


if __name__ == '__main__':
    unittest.main()
