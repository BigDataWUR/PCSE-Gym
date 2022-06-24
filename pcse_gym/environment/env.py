import datetime
import os

import numpy as np
import yaml

import gym

import pcse

"""
    OpenAI Gym Environment built around the PCSE library for crop simulation
    Gym:  https://github.com/openai/gym
    PCSE: https://github.com/ajwdewit/pcse
    
    Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/BigDataWUR/crop-gym)
    
"""


class Engine(pcse.engine.Engine):
    """
    Wraps around the PCSE engine/crop model to set a flag when the simulation has terminated
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flag_terminated = False

    @property
    def terminated(self):
        return self._flag_terminated

    def _terminate_simulation(self, day):
        super()._terminate_simulation(day)
        self._flag_terminated = True


class PCSEEnv(gym.Env):

    # TODO -- render modes for agent (render nothing) and humans (show plots of progression)

    _PATH_TO_FILE = os.path.dirname(os.path.realpath(__file__))
    _CONFIG_PATH = os.path.join(_PATH_TO_FILE, 'configs')

    _DEFAULT_AGRO_FILE = 'agromanagement_fertilization.yaml'
    _DEFAULT_CROP_FILE = 'lintul3_winterwheat.crop'
    _DEFAULT_SITE_FILE = 'lintul3_springwheat.site'
    _DEFAULT_SOIL_FILE = 'lintul3_springwheat.soil'

    _DEFAULT_AGRO_FILE_PATH = os.path.join(_CONFIG_PATH, 'agro', _DEFAULT_AGRO_FILE)
    _DEFAULT_CROP_FILE_PATH = os.path.join(_CONFIG_PATH, 'crop', _DEFAULT_CROP_FILE)
    _DEFAULT_SITE_FILE_PATH = os.path.join(_CONFIG_PATH, 'site', _DEFAULT_SITE_FILE)
    _DEFAULT_SOIL_FILE_PATH = os.path.join(_CONFIG_PATH, 'soil', _DEFAULT_SOIL_FILE)

    _DEFAULT_CONFIG = 'Lintul3.conf'

    # TODO -- logging?

    # TODO -- various pre-configured models

    # TODO possibility to change dates? Utility function to create agro management files?

    # TODO -- pass weather data provider as parameter?

    def __init__(self,
                 model_config: str =_DEFAULT_CONFIG,
                 agro_config: str =_DEFAULT_AGRO_FILE_PATH,
                 crop_parameters=_DEFAULT_CROP_FILE_PATH,
                 site_parameters=_DEFAULT_SITE_FILE_PATH,
                 soil_parameters=_DEFAULT_SOIL_FILE_PATH,
                 latitude: float = 52,
                 longitude: float = 5.5,  # TODO -- right values
                 seed: int = None,
                 timestep: int = 1,
                 batch_size: int = 1,  # TODO
                 ):
        assert timestep > 0
        assert batch_size > 0

        if isinstance(crop_parameters, str):
            crop_parameters = pcse.fileinput.PCSEFileReader(crop_parameters)
        if isinstance(site_parameters, str):
            site_parameters = pcse.fileinput.PCSEFileReader(site_parameters)
        if isinstance(soil_parameters, str):
            soil_parameters = pcse.fileinput.PCSEFileReader(soil_parameters)

        # Optionally set the seed
        if seed is not None:
            self.seed(seed)

        # Set location
        self._location = (latitude, longitude)
        self._timestep = timestep

        # Store the crop/soil/site parameters
        self._crop_params = crop_parameters
        self._site_params = site_parameters
        self._soil_params = soil_parameters

        # Store the agro-management config
        with open(agro_config, 'r') as f:
            self._agro_management = yaml.load(f, Loader=yaml.SafeLoader)

        # Store the PCSE Engine config
        self._model_config = model_config

        # Get the weather data source
        self._weather_data_provider = self._get_weather_data_provider()

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()

        # Use the config files to extract relevant settings
        model_config = pcse.util.ConfigurationLoader(model_config)
        self._output_variables = model_config.OUTPUT_VARS  # variables given by the PCSE model output
        self._summary_variables = model_config.SUMMARY_OUTPUT_VARS  # Summary variables are given at the end of a run
        self._weather_variables = list(pcse.base.weather.WeatherDataContainer.required)  # TODO -- configurable?

        # Define Gym observation space
        self.observation_space = self._get_observation_space()
        # Define Gym action space
        self.action_space = self._get_action_space()

    def _init_pcse_model(self, *args, **kwargs) -> Engine:

        # Combine the config files in a single PCSE ParameterProvider object
        self._parameter_provider = pcse.base.ParameterProvider(cropdata=self._crop_params,
                                                               sitedata=self._site_params,
                                                               soildata=self._soil_params,
                                                               )
        # Create a PCSE engine / crop growth model
        model = Engine(self._parameter_provider,
                       self._weather_data_provider,
                       self._agro_management,
                       config=self._model_config,
                       )
        # The model starts with output values for the initial date
        # The initial observation should contain output values for an entire timestep
        # If the timestep > 1, generate the remaining outputs by running the model
        if self._timestep > 1:
            model.run(days=self._timestep - 1)
        return model

    def _get_observation_space(self) -> gym.spaces.Space:   # TODO -- proper Box min/max values
        space = gym.spaces.Dict({
            'crop_model': self._get_observation_space_crop_model(),
            'weather': self._get_observation_space_weather(),
        })
        return space

    def _get_observation_space_weather(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                'IRRAD': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'TMIN': gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                'TMAX': gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                'VAP': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'RAIN': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'E0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'ES0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'ET0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'WIND': gym.spaces.Box(0, np.inf, (self._timestep,)),
            }
        )

    def _get_observation_space_crop_model(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {var: gym.spaces.Box(0, np.inf, shape=(self._timestep,)) for var in self._output_variables}
        )

    def _get_action_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict(
            {
                'irrigation': gym.spaces.Box(0, np.inf, shape=()),
                'N': gym.spaces.Box(0, np.inf, shape=()),
                'P': gym.spaces.Box(0, np.inf, shape=()),
                'K': gym.spaces.Box(0, np.inf, shape=()),
            }
        )
        return space  # TODO -- add more actions?

    def _get_weather_data_provider(self) -> pcse.db.NASAPowerWeatherDataProvider:
        wdp = pcse.db.NASAPowerWeatherDataProvider(*self._location)  # TODO -- other weather data providers
        return wdp

    """
    Properties of the crop model config file
    """

    @property
    def output_variables(self) -> list:
        return list(self._output_variables)

    @property
    def summary_variables(self) -> list:
        return list(self._summary_variables)

    @property
    def weather_variables(self):
        return list(self._weather_variables)

    """
    Properties derived from the agro management config:
    """

    @property
    def _campaigns(self) -> dict:
        return self._agro_management[0]

    @property
    def _first_campaign(self) -> dict:
        return self._campaigns[min(self._campaigns.keys())]

    @property
    def _last_campaign(self) -> dict:
        return self._campaigns[max(self._campaigns.keys())]

    @property
    def start_date(self) -> datetime.date:
        return self._model.agromanager.start_date

    @property
    def end_date(self) -> datetime.date:
        return self._model.agromanager.end_date

    """
    Other properties
    """

    @property
    def date(self) -> datetime.date:
        return self._model.day

    """
    Gym functions
    """

    def step(self, action) -> tuple:

        # Create a dict for storing info
        info = dict()

        # Apply action
        self._apply_action(action)

        # Run the crop growth model
        self._model.run(days=self._timestep)
        # Get the model output
        output = self._model.get_output()[-self._timestep:]
        info['days'] = [day['day'] for day in output]

        # Construct an observation and reward from the new environment state
        o = self._get_observation(output)
        r = self._get_reward(output)
        # Check whether the environment has terminated
        done = self._model.terminated
        if done:
            info['output_history'] = self._model.get_output()
            info['summary_output'] = self._model.get_summary_output()
            info['terminal_output'] = self._model.get_terminal_output()

        # Return all values
        return o, r, done, info

    def _apply_action(self, action):

        self._model._send_signal(signal=pcse.signals.irrigate,
                                 amount=action['irrigation'],
                                 efficiency=0.8,  # TODO -- what is a good value?
                                 )
        self._model._send_signal(signal=pcse.signals.apply_npk,
                                 N_amount=action['N'],
                                 P_amount=action['P'],
                                 K_amount=action['K'],
                                 N_recovery=0.7,
                                 P_recovery=0.7,
                                 K_recovery=0.7,  # TODO -- good values -- how to pass these to the function?
                                 )

    def _get_observation(self, output) -> dict:

        # Get the datetime objects characterizing the specific days
        days = [day['day'] for day in output]

        # Get the output variables for each of the days
        crop_model_observation = {v: [day[v] for day in output] for v in self._output_variables}

        # Get the weather data of the passed days
        weather_data = [self._weather_data_provider(day) for day in days]
        # Cast the weather data into a dict
        weather_observation = {var: [getattr(weather_data[d], var) for d in range(len(days))] for var in self._weather_variables}

        o = {
            'crop_model': crop_model_observation,
            'weather': weather_observation,
        }

        return o

    def _get_reward(self, _) -> float:
        output = self._model.get_output()
        # Consider different cases:
        if len(output) == 0:  # The simulation has not started -> 0 reward
            return 0
        if len(output) <= self._timestep:  # Only one observation is made -> give initial yield as reward
            return output[-1]['WSO']
        else:  # Multiple observations are made -> give difference of yield of the last time steps
            return output[-1]['WSO'] - output[-self._timestep - 1]['WSO']

    def reset(self,
              *,
              seed: int = None,
              return_info: bool = False,
              options: dict = None
              ):

        # Create an info dict
        info = dict()

        # Optionally set the seed
        if seed is not None:
            self.seed(seed)

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()

        output = self._model.get_output()[-1:]
        o = self._get_observation(output)
        info['date'] = self.date

        return o, info if return_info else o

    def render(self, mode="human"):
        pass  # Nothing to see here

    """
    Other functions
    """

    def save(self):
        # is it possible to save and restore pcse engine state?
        # Maybe pickle all the things?
        pass  # TODO

    @staticmethod
    def load():
        pass  # TODO

    # TODO -- save state/ load state?


if __name__ == '__main__':
    import time

    _env = PCSEEnv(timestep=1)
    _env.reset()

    print(_env.start_date)

    def _as_action(i, n, p, k):
        return {
            'irrigation': i,
            'N': n,
            'P': p,
            'K': k,
        }

    # _a = _as_action(1, 2, 3, 4)
    _a = _as_action(0, 0, 0, 0)

    _observations = []

    _d = False
    while not _d:
        # time.sleep(0.1)
        _o, _r, _d, _info = _env.step(_a)

        _observations += [{**_o['crop_model'], **_o['weather']}]

        print('\n'.join(
            [
                f'O: {_o}',
                f'R: {_r}',
                f'D: {_d}',
                f'I: {_info}',
            ]
        ))

    _output = _env._model.get_output()

    def mean(xs):
        return sum(xs) / len(xs)

    import math
    def std(xs):
        return math.sqrt(sum([(x - sum(xs)) ** 2 for x in xs]) / len(xs))

    _means = [mean([day[_var][0] for day in _observations]) for _var in _env.output_variables + _env.weather_variables]
    _stds = [std([day[_var][0] for day in _observations]) for _var in _env.output_variables + _env.weather_variables]

    # print(_means)
    print(_stds)