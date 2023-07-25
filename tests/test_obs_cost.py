import unittest
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import *
from pcse_gym.utils.defaults import *
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def initialize_env():
    pcse_env = 1
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
                                locations=get_default_location(),
                                action_space=action_spaces, action_multiplier=1.0, reward='DEF',
                                **get_model_kwargs(pcse_env), **kwargs)

    return env_pcse_eval


def initialize_no_baseline():
    action_spaces = gym.spaces.Discrete(7)

    env_pcse_eval = WinterWheat(crop_features=get_wofost_default_crop_features(),
                                action_features=get_default_action_features(),
                                weather_features=get_default_weather_features(),
                                costs_nitrogen=10, years=get_default_train_years(),
                                locations=get_default_location(),
                                action_space=action_spaces, action_multiplier=2.0, reward='GRO',
                                **get_model_kwargs(1))

    return env_pcse_eval


def initialize_random_env():
    pcse_env = 1
    if pcse_env:
        po_features = ['TAGP', 'LAI', 'NAVAIL', 'SM', 'NuptakeTotal', 'random']
        crop_features = ["DVS", "TAGP", "LAI", "NuptakeTotal", "NAVAIL", "SM"]
        if 'random' in po_features:
            crop_features.append('random')
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
                                locations=get_default_location(),
                                action_space=action_spaces, action_multiplier=1.0, reward='DEF',
                                **get_model_kwargs(pcse_env), **kwargs)

    return env_pcse_eval


env = initialize_env()

env2 = initialize_no_baseline()

env3 = initialize_random_env()

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

        self.assertListEqual(expected, actual)


class TestNoMeasure(unittest.TestCase):

    def test_output(self):
        env2.reset()
        action = np.array([3])
        _, _, _, _, _ = env2.step(action)
        action = np.array([4])
        obs, reward, terminated, truncated, info = env2.step(action)


class TestRandomFeature(unittest.TestCase):

    def test_random(self):
        env3.reset()
        action = np.array([0, 1, 0, 0, 0, 1, 1])
        measure = action[1:]
        obs, reward, terminated, truncated, info = env3.step(action)
        expected = []
        costs = env3.measure_features.get_observation_cost()
        for i, cost in zip(measure, costs):
            if i:
                expected.append(cost)
            else:
                expected.append(0)
        #expected = env3.measure_features.list_of_costs()
        expected = -float(sum(expected))

        self.assertEqual(expected, reward)


# TODO add unit test CERES

if __name__ == '__main__':
    unittest.main()
