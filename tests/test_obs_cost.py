import unittest
import numpy as np

import tests.initialize_env as init_env


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_po()

    def test_oc(self):
        self.env.reset()
        cost = self.env.measure_features.get_observation_cost()

        obs = np.ones(30)

        measure = [1, 1, 1, 1, 1]

        _, measurement_cost = self.env.measure_features.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))

    def test_index_match_with_obs(self):
        self.env.reset()
        action = np.array([0, 1, 0, 0, 0, 1])

        obs, reward, terminated, truncated, info = self.env.step(action)

        actual = []
        for i in range(1, 4):
            a = obs[self.env.measure_features.feature_ind[i]]
            actual.append(a)
        expected = list(np.zeros(3))
        self.assertListEqual(expected, actual)


class TestNoMeasure(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_no_baseline()

    def test_output(self):

        self.env.reset()
        action = np.array([3])
        _, _, _, _, _ = self.env.step(action)
        action = np.array([4])
        obs, reward, terminated, truncated, info = self.env.step(action)

        expected = -80
        actual = reward

        self.assertEqual(expected, actual)


class TestRandomFeature(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_random()

    def test_random(self):
        self.env.reset()
        action = np.array([0, 1, 0, 0, 0, 1, 1])
        measure = action[1:]
        obs, reward, terminated, truncated, info = self.env.step(action)
        expected = []
        costs = self.env.measure_features.get_observation_cost()
        for i, cost in zip(measure, costs):
            if i:
                expected.append(cost)
            else:
                expected.append(0)
        expected = -float(sum(expected))

        self.assertEqual(expected, reward)
