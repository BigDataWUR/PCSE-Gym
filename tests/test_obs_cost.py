import unittest
import numpy as np

import tests.initialize_env as init_env


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_po()
        self.env_ext = init_env.initialize_env_measure_po_extend()
        self.env_no_cost = init_env.initialize_env_measure_no_cost()
        self.env_same_cost = init_env.initialize_env_measure_same_cost()
        self.env_multiplier = init_env.initialize_env_multiplier()

    def test_oc(self):
        self.env.reset()

        obs = np.ones(30)

        measure = [1, 1, 1, 1, 1]

        cost = []
        for m, f in zip(measure, self.env.measure_features.feature_ind):
            cost.append(self.env.measure_features.get_observation_cost(m,f))

        _, measurement_cost = self.env.measure_features.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))

    def test_no_oc(self):

        self.env_no_cost.reset()

        obs = np.ones(30)

        measure = [1, 1, 1, 1, 1]

        cost = []
        for m, f in zip(measure, self.env_no_cost.measure_features.feature_ind):
            cost.append(self.env_no_cost.measure_features.get_observation_cost(m, f))

        _, measurement_cost = self.env_no_cost.measure_features.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))

    def test_same_oc(self):

        self.env_same_cost.reset()

        obs = np.ones(30)

        measure = [1, 1, 1, 1, 1]

        cost = []
        for m, f in zip(measure, self.env_same_cost.measure_features.feature_ind):
            cost.append(self.env_same_cost.measure_features.get_observation_cost(m, f))

        _, measurement_cost = self.env_same_cost.measure_features.measure_act(obs, measure)

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

    def test_mask_obs(self):
        self.env_ext.reset()
        action = np.array([0, 1, 0, 0, 0, 1])

        obs, reward, terminated, truncated, info = self.env_ext.step(action)

        obs_obs = obs[:6]

        obs_mask = np.array([1.0, 0.0, 1.0, 0.0, 0.0])

        expected_obs = np.append(obs_obs, obs_mask)

        self.assertListEqual(expected_obs.tolist(), obs.tolist())

    def test_obs_cost_multiplier(self):
        self.env_multiplier.reset()
        action = np.array([0, 1, 1, 1, 1, 1])

        _, reward, _, _, _ = self.env_multiplier.step(action)

        self.assertEqual( -150, reward)


class TestNoisyMeasure(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_po_noisy()

    def test_noisy_oc(self):
        self.env.reset()
        obs = np.ones(30)
        measure = [1, 2, 1, 2, 1]
        cost = []
        for m, f in zip(measure, self.env.measure_features.feature_ind):
            cost.append(self.env.measure_features.get_observation_cost(m, f))
        _, measurement_cost = self.env.measure_features.measure_act(obs, measure)

        self.assertListEqual(cost, list(measurement_cost))

    def test_noise_value(self):
        self.env.reset()
        action = np.array([0, 2, 2, 2, 2, 2])
        obs, _, _, _, _ = self.env.step(action)

        lai_std, sm_std, navail_std, nuptaketotal_std, tagp_std = 0.4, 0.2, 5, 5, 2

        lai = obs[self.env.measure_features.feature_ind_dict['LAI']]
        sm = obs[self.env.measure_features.feature_ind_dict['SM']]
        navail = obs[self.env.measure_features.feature_ind_dict['NAVAIL']]
        nuptaketotal = obs[self.env.measure_features.feature_ind_dict['NuptakeTotal']]
        tagp = obs[self.env.measure_features.feature_ind_dict['TAGP']]

        # assert that range of noisy observations are within the defined std
        self.assertLessEqual(lai, lai + lai_std)
        self.assertGreaterEqual(lai, lai - lai_std)
        self.assertLessEqual(sm, sm + sm_std)
        self.assertGreaterEqual(sm, sm - sm_std)
        self.assertLessEqual(navail, navail + navail_std)
        self.assertGreaterEqual(navail, navail - navail_std)
        self.assertLessEqual(nuptaketotal, nuptaketotal + nuptaketotal_std)
        self.assertGreaterEqual(nuptaketotal, nuptaketotal - nuptaketotal_std)
        self.assertLessEqual(tagp, tagp + tagp_std)
        self.assertGreaterEqual(tagp, tagp - tagp_std)


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


class TestNonSelective(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_non_selective()

    def test_cost(self):

        self.env.reset()
        action = np.array([0, 1])
        _, reward, _, _, _ = self.env.step(action)

        expected = -1

        self.assertEqual(expected, reward)

        action = np.array([4, 1])
        obs, reward, terminated, truncated, info = self.env.step(action)

        expected = -40 - 1

        self.assertEqual(expected, reward)

    def test_obs(self):

        self.env.reset()
        action = np.array([5, 0])
        obs, reward, _, _, _ = self.env.step(action)

        expected = np.empty(5)
        expected.fill(-1.11)

        self.assertListEqual(list(expected), list(obs[1:6]))


class TestRandomFeature(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_random()

    def test_random(self):
        self.env.reset()
        action = np.array([0, 1, 0, 0, 0, 1, 1])
        measure = action[1:]
        obs, reward, terminated, truncated, info = self.env.step(action)
        expected = []
        for num, m in zip(self.env.measure_features.feature_ind, measure):
            if m:
                expected.append(self.env.measure_features.get_observation_cost(m, num))
            else:
                expected.append(0)
        expected = -float(sum(expected))

        self.assertEqual(expected, reward)
