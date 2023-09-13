import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy

from ray.rllib.utils.typing import PolicyID

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker

import pcse_gym.utils.defaults as defaults
import pcse_gym.utils.eval as eval
from pcse_gym.envs.winterwheat import WinterWheatRay
from pcse_gym.envs.constraints import ActionConstrainer


def ww_unwrapped_unnormalized(env_config):
    return WinterWheatRay(env_config)


def ww_nor(env_config):
    env = WinterWheatRay(env_config)
    env = NormalizeObservation(NormalizeReward(env))
    return env


def ww_lim(env_config):
    env = WinterWheatRay(env_config)
    action_limit = env_config.get("action_limit", 0)
    n_budget = env_config.get("n_budget", 0)
    env = ActionConstrainer(env, action_limit=action_limit, n_budget=n_budget)
    return env


def ww_lim_norm(env_config):
    env = WinterWheatRay(env_config)
    action_limit = env_config.get("action_limit", 0)
    n_budget = env_config.get("n_budget", 0)
    env = ActionConstrainer(env, action_limit=action_limit, n_budget=n_budget)
    env = NormalizeObservation(NormalizeReward(env))
    return env


def winterwheat_config_maker(crop_features=defaults.get_wofost_default_crop_features(),
                             action_features=defaults.get_default_action_features(),
                             weather_features=defaults.get_default_weather_features(),
                             seed=0, costs_nitrogen=None, timestep=7, years=None, locations=None,
                             action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
                             action_multiplier=1.0, reward=None,
                             *args, **kwargs):
    config = dict(
        crop_features=crop_features,
        action_features=action_features,
        weather_features=weather_features,
        seed=seed,
        costs_nitrogen=costs_nitrogen,
        timestep=timestep,
        years=years,
        locations=locations,
        action_space=action_space,
        action_multiplier=action_multiplier,
        reward=reward,
        args=args,
        kwargs=kwargs,
        action_limit=kwargs.get('action_limit', 0),
        n_budget=kwargs.get('n_budget', 0)
    )
    return config


def get_rllib_config(model, env_config, env="WinterWheatRay", action_limit=0, n_budget=0):
    bptt_size = 1024
    return {
        "model": {
            "max_seq_len": bptt_size,
            "custom_model": model,
            "custom_model_config": {
                # Override the hidden_size from BASE_CONFIG
                # The input and output sizes of the MLP feeding the memory model
                "preprocessor_input_size": 128,
                "preprocessor_output_size": 64,
                "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                # this is the size of the recurrent state in most cases
                "hidden_size": 128,
                # We should also change other parts of the architecture to use
                # this new hidden size
                # For the GRU, the output is of size hidden_size
                "postprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "postprocessor_output_size": 64,
                # Actor and critic networks
                "actor": nn.Linear(64, 64),
                "critic": nn.Linear(64, 64),
                # We can also override GRU-specific hyperparams
                "num_recurrent_layers": 1,
            },
        },
        # Some other rllib defaults you might want to change
        # See https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
        # for a full list of rllib settings
        #
        # These should be a factor of bptt_size
        "sgd_minibatch_size": bptt_size * 4,
        # Should be a factor of sgd_minibatch_size
        "train_batch_size": bptt_size * 8,
        # You probably don't want to change these values
        "rollout_fragment_length": bptt_size,
        "framework": "torch",
        "horizon": bptt_size,
        "batch_mode": "complete_episodes",

        # The environment we are training on
        "env": env,
        "env_config": env_config,
        # "callbacks": SB3CallbackWrapper
        # "disable_env_checking": True
    }


# TODO: implement callbacks with the RLlib tune.run training possibly mimicking implemented SB3 callback
class RayEvalCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: Optional[int] = None,
            **kwargs,
    ):
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env.reset()."
        )

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Union[Episode, EpisodeV2],
            env_index: Optional[int] = None,
            **kwargs,
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env.reset()"
        )

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ):
        print(episode)

    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        pass




