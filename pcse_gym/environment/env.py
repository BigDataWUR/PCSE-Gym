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

    def __init__(self,
                 model_config_file: str = _DEFAULT_CONFIG,
                 agro_config_file: str = _DEFAULT_AGRO_FILE_PATH,
                 crop_config_file: str = _DEFAULT_CROP_FILE_PATH,
                 site_config_file: str = _DEFAULT_SITE_FILE_PATH,
                 soil_config_file: str = _DEFAULT_SOIL_FILE_PATH,
                 latitude: float = 52,
                 longitude: float = 5.5,  # TODO -- right values
                 seed: int = None,
                 ):

        # Optionally set the seed
        if seed is not None:
            self.seed(seed)

        # Set location
        self._location = (latitude, longitude)

        # Load the PCSE Engine config
        model_config = pcse.util.ConfigurationLoader(model_config_file)
        # Store model config file for re-initialization
        self._model_config_file = model_config_file
        # Store which variables are given by the PCSE model output
        self._output_variables = model_config.OUTPUT_VARS
        self._summary_variables = model_config.SUMMARY_OUTPUT_VARS  # Summary variables are given at the end of a run

        # Read agro config file
        with open(agro_config_file, 'r') as f:
            self._agro_management = yaml.load(f, Loader=yaml.SafeLoader)  # TODO possibility to change dates? Utility function to create agro management files?

        # Read all other config files
        crop_config = pcse.fileinput.PCSEFileReader(crop_config_file)
        site_config = pcse.fileinput.PCSEFileReader(site_config_file)
        soil_config = pcse.fileinput.PCSEFileReader(soil_config_file)

        # Combine the config files in a single PCSE ParameterProvider object
        self._parameter_provider = pcse.base.ParameterProvider(cropdata=crop_config,
                                                               sitedata=site_config,
                                                               soildata=soil_config,
                                                               )

        # Get the weather data source
        self._weather_data_provider, self._weather_variables = self._get_weather_data_provider()

        # Create a PCSE engine / crop growth model
        self._model = Engine(self._parameter_provider,
                             self._weather_data_provider,
                             self._agro_management,
                             config=model_config_file,
                             )

        # Define Gym observation space
        self.observation_space = self._get_observation_space()
        # Define Gym action space
        self.action_space = self._get_action_space()

    def _get_observation_space(self) -> gym.spaces.Space:   # TODO -- proper Box min/max values
        space = gym.spaces.Dict({
            'crop_model': self._get_observation_space_crop_model(),
            'weather': self._get_observation_space_weather(),
        })
        return space

    def _get_observation_space_weather(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                'IRRAD': gym.spaces.Box(0, np.inf, ()),
                'TMIN': gym.spaces.Box(-np.inf, np.inf, ()),
                'TMAX': gym.spaces.Box(-np.inf, np.inf, ()),
                'VAP': gym.spaces.Box(0, np.inf, ()),
                'RAIN': gym.spaces.Box(0, np.inf, ()),
                'E0': gym.spaces.Box(0, np.inf, ()),
                'ES0': gym.spaces.Box(0, np.inf, ()),
                'ET0': gym.spaces.Box(0, np.inf, ()),
                'WIND': gym.spaces.Box(0, np.inf, ()),
            }
        )

    def _get_observation_space_crop_model(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {var: gym.spaces.Box(0, np.inf, shape=()) for var in self._output_variables}
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

    def _get_weather_data_provider(self) -> tuple:
        wdp = pcse.db.NASAPowerWeatherDataProvider(*self._location)  # TODO -- other weather data providers
        variables = list(pcse.base.weather.WeatherDataContainer.required)
        return wdp, variables

    """
    Properties of the crop model config file
    """

    @property
    def output_variables(self) -> list:
        return list(self._output_variables)

    @property
    def summary_variables(self) -> list:
        return list(self._summary_variables)

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
        self._model.run(days=1)  # TODO -- variable nr of days
        # Get the model output
        # TODO -- should this be considered a state? It is much closer to an observation/partial state
        output = self._model.get_output()
        state = output[-1]
        info['date'] = state['day']
        # Construct an observation and reward from the new environment state
        o = self._get_observation(state)
        r = self._get_reward(state)
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

    def _get_observation(self, state) -> dict:

        o = {v: state[v] for v in self.output_variables}

        weather_data = self._weather_data_provider(self._model.day)
        weather_observation = {k: getattr(weather_data, k) for k in self._weather_variables}

        o['weather'] = weather_observation

        return o

    def _get_reward(self, state) -> float:
        output = self._model.get_output()
        # Consider different cases:
        if len(output) == 0:  # The simulation has not started -> 0 reward
            return 0
        if len(output) == 1:  # Only one observation is made -> give initial yield as reward
            return output[0]['WSO']
        else:  # Multiple observations are made -> give difference of yield of the last time steps
            return output[-1]['WSO'] - output[-2]['WSO']

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
        self._model = Engine(self._parameter_provider,
                             self._weather_data_provider,
                             self._agro_management,
                             config=self._model_config_file,
                             )

        state = self._model.get_output()[-1]
        o = self._get_observation(state)
        info['date'] = state['day']

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

    _env = PCSEEnv()

    print(_env.start_date)

    def _as_action(i, n, p, k):
        return {
            'irrigation': i,
            'N': n,
            'P': p,
            'K': k,
        }

    _a = _as_action(1, 2, 3, 4)

    _d = False
    while not _d:
        time.sleep(0.1)
        _o, _r, _d, _info = _env.step(_a)

        print('\n'.join(
            [
                f'O: {_o}',
                f'R: {_r}',
                f'D: {_d}',
                f'I: {_info}',
            ]
        ))




