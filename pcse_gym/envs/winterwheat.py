import datetime
import gymnasium as gym
import numpy as np

import pcse

import pcse_gym.envs.common_env as common_env
import pcse_gym.utils.defaults as defaults
import pcse_gym.utils.process_pcse_output as process_pcse
from pcse_gym.utils.normalization import NormalizeMeasureObservations, RunningReward, MinMaxReward
from .constraints import VariableRecoveryRate
from .measure import MeasureOrNot
from .sb3 import ZeroNitrogenEnvStorage, StableBaselinesWrapper
from .rewards import Rewards, DummyClass
from .rewards import reward_functions_without_baseline, reward_functions_end, calculate_nue, get_surplus_n


class WinterWheat(gym.Env):
    """
    Environment with two sub-environments:
        (1) environment for applying actions of RL agent
        (2) a baseline environment (e.g. with zero nitrogen policy) for computing relative reward
    Year and location of episode is randomly picked from years and locations through reset().
    """

    def __init__(self, crop_features=defaults.get_wofost_default_crop_features(),
                 action_features=defaults.get_default_action_features(),
                 weather_features=defaults.get_default_weather_features(),
                 seed=0, costs_nitrogen=None, timestep=7, years=None, locations=None,
                 action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                 action_multiplier=1.0, reward=None,
                 *args, **kwargs
                 ):
        self.crop_features = crop_features
        self.action_features = action_features
        self.weather_features = weather_features
        self.costs_nitrogen = costs_nitrogen
        self.years = [years] if isinstance(years, int) else years
        self.locations = [locations] if isinstance(locations, tuple) else locations
        self.action_multiplier = action_multiplier
        self.action_space = action_space
        self._timestep = timestep
        self.reward_function = reward
        self.po_features = kwargs.get('po_features', [])
        self.mask_binary = kwargs.get('mask_binary', False)
        self.placeholder_val = kwargs.get('placeholder_val', -1.11)
        self.no_weather = kwargs.get('no_weather', False)
        self.normalize = kwargs.get('normalize', False)
        self.cost_measure = kwargs.get('cost_measure', 'real')
        self.random_init = kwargs.get('random_init', False)
        self.measure_cost_multiplier = kwargs.get('m_multiplier', 1)
        self.measure_all = kwargs.get('measure_all', False)
        self.list_wav_nav = None
        if self.random_init:
            self.eval_wav = None
            self.eval_nav = None
            self.list_wav_nav = [self.eval_wav, self.eval_nav]
        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)

        if self.reward_function not in reward_functions_without_baseline():
            self._env_baseline = self._initialize_sb_wrapper(seed, *args, **kwargs)
        self._env = self._initialize_sb_wrapper(seed, *args, **kwargs)

        self.observation_space = self._get_observation_space()
        self.zero_nitrogen_env_storage = ZeroNitrogenEnvStorage()

        # Reward Init

        self._init_reward_function(costs_nitrogen, kwargs)

        if self.po_features:
            self.__measure = MeasureOrNot(self.sb3_env, extend_obs=self.mask_binary,
                                          placeholder_val=self.placeholder_val,
                                          cost_multiplier=self.measure_cost_multiplier,
                                          measure_all_flag=self.measure_all)

        if self.normalize:
            self.loc_code = kwargs.get('loc_code', None)
            self._norm = NormalizeMeasureObservations(self.crop_features, self.measure_features.feature_ind,
                                                      has_random=True if 'random' in self.crop_features else False,
                                                      no_weather=self.no_weather, loc=self.loc_code,
                                                      start_type=kwargs.get('start_type', 'sowing'),
                                                      mask_binary=self.mask_binary, reward_div=600, is_clipped=False)
            # self._rew_norm = MinMaxReward()

        super().reset(seed=seed)

    def _init_reward_function(self, costs_nitrogen, kwargs):

        self.rewards_obj = Rewards(kwargs.get('reward_var'), self.timestep, costs_nitrogen)
        self.reward_container = DummyClass()

        if self.reward_function == 'ANE':
            self.rewards = self.rewards_obj.DEF(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerANE(self.timestep)

        elif self.reward_function == 'DEF':
            self.rewards = self.rewards_obj.DEF(self.timestep, costs_nitrogen)

        elif self.reward_function == 'GRO':
            self.rewards = self.rewards_obj.GRO(self.timestep, costs_nitrogen)

        elif self.reward_function == 'DEP':
            self.rewards = self.rewards_obj.DEP(self.timestep, costs_nitrogen)

        elif self.reward_function in reward_functions_end():
            self.rewards = self.rewards_obj.END(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerEND(self.timestep, costs_nitrogen)

        elif self.reward_function == 'NUE':
            self.rewards = self.rewards_obj.NUE(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)

        else:
            self.rewards = self.rewards_obj.DEF(self.timestep, costs_nitrogen)

    def _initialize_sb_wrapper(self, seed, *args, **kwargs):
        return StableBaselinesWrapper(crop_features=self.crop_features,
                                      action_features=self.action_features,
                                      weather_features=self.weather_features,
                                      costs_nitrogen=self.costs_nitrogen,
                                      timestep=self.timestep,
                                      years=self.years[0], location=self.locations[0],
                                      action_space=self.action_space,
                                      action_multiplier=self.action_multiplier,
                                      seed=seed,
                                      *args, **kwargs)

    def _get_observation_space(self):
        nvars = self._get_obs_len()
        return gym.spaces.Box(-10, np.inf, shape=(nvars,))

    def _get_obs_len(self):
        if self.sb3_env.no_weather:
            nvars = len(self.crop_features)
        else:
            nvars = len(self.crop_features) + len(self.weather_features) * self.timestep
        if self.mask_binary:  # TODO: test with weather features
            nvars = nvars + len(self.po_features)
        return nvars

    def step(self, action):
        """
        Computes customized reward and populates info
        """

        # advance one step of the PCSEEngine wrapper and apply action(s)
        obs, _, terminated, truncated, info = self.sb3_env.step(action)

        # grab output of PCSE
        output = self.sb3_env.model.get_output()

        # process output to get observation, reward and growth of winterwheat
        obs, reward, growth = self.process_output(action, output, obs)

        # fill in infos
        if 'reward' not in info.keys(): info['reward'] = {}
        info['reward'][self.date] = reward
        if 'growth' not in info.keys(): info['growth'] = {}
        info['growth'][self.date] = growth

        # used for reward functions that rely on rewards at terminate
        reward, info = self.end_reward_modifiers(output, reward, terminated, info)

        # normalize observations and reward if not using VecNormalize wrapper
        if self.normalize:
            measure = None
            if isinstance(action, np.ndarray):
                measure = action[1:]
            obs = self.norm.normalize_measure_obs(obs, measure)
            self.norm.update_running_rew(reward)
            reward = self.norm.normalize_reward(reward)

        return obs, reward, terminated, truncated, info

    def process_output(self, action, output, obs):

        if self.po_features and isinstance(action, np.ndarray) and action.dtype != np.float32:
            measure = None
            if isinstance(action, np.ndarray):
                action, measure = action[0], action[1:]
            amount = action * self.action_multiplier
            reward, growth = self.get_reward_and_growth(output, amount)
            obs, cost = self.measure_features.measure_act(obs, measure)
            measurement_cost = sum(cost)
            # if self.reward_function in reward_functions_end() and self.reward_function == 'NUE':
            #     self.rewards.calc_misc_cost(self.reward_container, measurement_cost)
            reward -= measurement_cost
            return obs, reward, growth
        else:
            if isinstance(action, np.ndarray):
                action = action.item()
            amount = action * self.action_multiplier
            reward, growth = self.get_reward_and_growth(output, amount)
            return obs, reward, growth

    def get_reward_and_growth(self, output, amount):
        output_baseline = []
        if self.reward_function not in reward_functions_without_baseline():
            zero_nitrogen_results = self.zero_nitrogen_env_storage.get_episode_output(self.baseline_env)
            # convert zero_nitrogen_results to pcse_output
            var_name = process_pcse.get_name_storage_organ(zero_nitrogen_results.keys())
            for (k, v) in zero_nitrogen_results[var_name].items():
                if k <= output[-1]['day']:
                    filtered_dict = {'day': k, var_name: v}
                    output_baseline.append(filtered_dict)

        reward, growth = self.rewards.return_reward(output, amount,
                                                    output_baseline=output_baseline,
                                                    multiplier=self.sb3_env.multiplier_amount,
                                                    obj=self.reward_container)
        return reward, growth

    def end_reward_modifiers(self, output, reward, terminated, info):
        if terminated and self.reward_function in reward_functions_end():
            reward = self.reward_container.dump_cumulative_positive_reward - abs(reward)

        elif terminated and self.reward_function == 'NUE':
            n_so = output[-1]['NamountSO']
            cum_actions = self.reward_container.actions * 10
            reward = self.reward_container.calculate_reward_nue(cum_actions, n_so, year=self.date.year)
            if 'NUE' not in info.keys():
                info['NUE'] = {}
            info['NUE'][self.date] = calculate_nue(cum_actions, n_so, year=self.date.year)
            if 'Nsurplus' not in info.keys():
                info['Nsurplus'] = {}
            info['Nsurplus'][self.date] = get_surplus_n(cum_actions, n_so, year=self.date.year)

        return reward, info

    def overwrite_year(self, year):
        self.years = year
        if self.reward_function not in reward_functions_without_baseline():
            self.baseline_env.agro_management = self.sb3_env.agmt.replace_years(year)
        self.sb3_env.agro_management = self.sb3_env.agmt.replace_years(year)

    def set_location(self, location):
        if self.reward_function not in reward_functions_without_baseline():
            self.baseline_env.loc = location
            self.baseline_env.weather_data_provider = common_env.get_weather_data_provider(location)
        self.sb3_env.loc = location
        self.sb3_env.weather_data_provider = common_env.get_weather_data_provider(location)

    def overwrite_location(self, location):
        self.locations = location
        self.set_location(location)

    def overwrite_initial_conditions(self):
        # method to overwrite a random initial condition for every call of reset()
        wav = np.clip(self.rng.normal(15, 15), 0.0, 100.0)
        nav = np.clip(self.rng.normal(15, 15), 0.0, 100.0)
        self.eval_wav = wav
        self.eval_nav = nav
        site_parameters=pcse.util.WOFOST80SiteDataProvider(WAV=wav, NAVAILI=nav)
        return site_parameters

    def reset(self, seed=None, options=None, **kwargs):
        site_params = None
        if self.random_init:
            site_params = self.overwrite_initial_conditions()

        if isinstance(self.years, list):
            year = self.np_random.choice(self.years)
            if self.reward_function not in reward_functions_without_baseline():
                self.baseline_env.agro_management = self.sb3_env.agmt.replace_years(year)
            self.sb3_env.agro_management = self.sb3_env.agmt.replace_years(year)

        if isinstance(self.locations, list):
            location = self.locations[self.np_random.choice(len(self.locations), 1)[0]]
            self.set_location(location)

        self.reward_container.reset()

        if self.reward_function not in reward_functions_without_baseline():
            self.baseline_env.reset(seed=seed, options=site_params)
        obs = self.sb3_env.reset(seed=seed, options=site_params)

        # TODO: check whether info should/could be filled
        info = {}

        if self.normalize:
            obs = self.norm.normalize_measure_obs(obs, None)

        return obs, info

    def render(self, mode="human"):
        pass

    @property
    def measure_features(self):
        return self.__measure

    @property
    def model(self):
        return self.sb3_env.model

    @model.setter
    def model(self, model):
        self.sb3_env.model = model

    @property
    def norm(self):
        return self._norm

    @property
    def norm_rew(self):
        return self._rew_norm

    @property
    def sb3_env(self):
        return self._env

    @property
    def baseline_env(self):
        return self._env_baseline

    @property
    def date(self) -> datetime.date:
        return self.sb3_env.model.day

    @property
    def loc(self) -> datetime.date:
        return self.sb3_env.loc

    @property
    def timestep(self):
        return self._timestep

    @property
    def obs_len(self):
        return self._get_obs_len()

    @property
    def act_len(self):
        return len(self.action_space.shape)


class WinterWheatRay(WinterWheat):
    """
    A wrapper for Ray's RLlib that replaces most of the param input arguments with a config file containing all the info

    @:param config: a config file or a dict that contains input parameters for the custom environment; in this case
    winterwheat
    """
    def __init__(self, config, *args, **kwargs):
        super(WinterWheatRay, self).__init__(crop_features=config['crop_features'],
                                             action_features=config['action_features'],
                                             weather_features=config['weather_features'],
                                             seed=config['seed'],
                                             costs_nitrogen=config['costs_nitrogen'],
                                             timestep=config['timestep'],
                                             years=config['years'], locations=config['locations'],
                                             action_space=config['action_space'],
                                             action_multiplier=config['action_multiplier'],
                                             reward=config['reward'], *config['args'],
                                             **config['kwargs'])
        self.pcse_model = config['pcse_model']
