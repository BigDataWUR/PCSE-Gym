import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, List

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from ray.rllib import SampleBatch

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy

from ray.rllib.utils.typing import PolicyID, AgentID

from ray.air.integrations.comet import CometLoggerCallback

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker

import pcse_gym.utils.defaults as defaults
import pcse_gym.utils.eval as eval
from pcse_gym.envs.winterwheat import WinterWheatRay
from pcse_gym.envs.constraints import ActionConstrainer

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from popgym.baselines.ray_models.ray_mlp import MLP
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.baselines.ray_models.ray_indrnn import IndRNN
from popgym.baselines.ray_models.ray_diffnc import DiffNC
from popgym.baselines.ray_models.ray_s4d import S4D


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
                             observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                             action_multiplier=1.0, reward=None, pcse_model=1, eval_years=None,
                             eval_locations=None,
                             *args, **kwargs):
    config = dict(
        crop_features=crop_features,
        action_features=action_features,
        weather_features=weather_features,
        seed=seed,
        costs_nitrogen=costs_nitrogen,
        timestep=timestep,
        years=years,
        eval_years=eval_years,
        locations=locations,
        eval_locations=eval_locations,
        action_space=action_space,
        observation_space=observation_space,
        action_multiplier=action_multiplier,
        reward=reward,
        args=args,
        kwargs=kwargs,
        action_limit=kwargs.get('action_limit', 0),
        n_budget=kwargs.get('n_budget', 0),
        pcse_model=pcse_model,

    )
    return config


def get_algo(model):
    if model == 'GRU':
        return GRU
    if model == 'PosMLP':
        return MLP
    if model == 'IndRNN':
        return IndRNN
    if model == 'DiffNC':
        return DiffNC
    if model == 'S4D':
        return S4D
    else:
        raise Exception("Agent name error!")


def modify_algo_config(conf, model):
    conf['model']['custom_model_config']["preprocessor_input_size"] = 256
    conf['model']['custom_model_config']["preprocessor"] = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
    conf['model']['custom_model_config']["preprocessor_output_size"] = 256
    conf['model']['custom_model_config']["hidden_size"] = 256
    conf['model']['custom_model_config']["postprocessor"] = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
    conf['model']['custom_model_config']["postprocessor_output_size"] = 256
    conf['model']['custom_model_config']["actor"] = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
    conf['model']['custom_model_config']["critic"] = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
    conf['model']['custom_model_config']["postprocessor_output_size"] = 256
    if model == 'GRU':
        # conf['model']['custom_model_config']["preprocessor_input_size"] = 128
        # conf['model']['custom_model_config']["preprocessor"] = nn.Sequential(nn.Linear(128, 256), nn.LeakyReLU())
        # conf['model']['custom_model_config']["preprocessor_output_size"] = 256
        # conf['model']['custom_model_config']["hidden_size"] = 256
        # conf['model']['custom_model_config']["postprocessor"] = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU())
        # conf['model']['custom_model_config']["postprocessor_output_size"] = 256
        # conf['model']['custom_model_config']["actor"] = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU())
        # conf['model']['custom_model_config']["critic"] = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU())
        # conf['model']['custom_model_config']["postprocessor_output_size"] = 256
        conf['model']['custom_model_config']["num_recurrent_layers"] = 1
    if model == 'IndRNN':
        conf['model']['custom_model_config']["activation"] = "tanh"
    if model == 'DiffNC':
        conf['model']['custom_model_config']["num_hidden_layers"] = 1
        conf['model']['custom_model_config']["num_layers"] = 1
        conf['model']['custom_model_config']["read_heads"] = 4
        conf['model']['custom_model_config']["cell_size"] = 32
        conf['model']['custom_model_config']["nonlinearity"] = "tanh"
    return conf


def get_algo_config(model, env_config, env="WinterWheatRay"):
    bptt_size = 1024
    return {
        "model": {
            "max_seq_len": bptt_size,
            "custom_model": model,
            # "kl_coeff": 0.3,
            "custom_model_config": {
                ## Generally will be replaced
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
                "actor": nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU()),
                "critic": nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU()),
            },
        },
        # Some other rllib defaults you might want to change
        # See https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
        # for a full list of rllib settings
        #
        # These should be a factor of bptt_size
        "sgd_minibatch_size": bptt_size * 4,
        # Should be a factor of sgd_minibatch_size
        "train_batch_size": bptt_size * 8,  # size has to be (rollout_fragment_length*num_num_rollout_workers*
        # num_envs_per_worker)

        'lr': 0.0001,

        # You probably don't want to change these values
        "rollout_fragment_length": bptt_size,
        "framework": "torch",
        "horizon": bptt_size,
        "batch_mode": "complete_episodes",

        # The environment we are training on
        "env": env,
        "env_config": env_config,
        "callbacks": RayEvalCallback,
        # "disable_env_checking": True,
        "keep_per_episode_custom_metrics": True,

        # resources
        "num_rollout_workers": 8,
        "num_envs_per_worker": 1,  # if more than one, possibility of sync error with PCSE

        # other stuff
        # "_enable_rl_module_api": False,
        # "_enable_learner_api": False,
    }


# TODO: document how to install ray[rllib] to make this operational
class RayEvalCallback(DefaultCallbacks):
    """
    Callback wrapper to run external test evaluations of the trained agent everytime
    a training iteration is finished
    """

    def __init__(self):
        super().__init__()

    def evaluate_rllib_policy(self, policy, env, n_eval_episodes=1):
        episode_rewards, episode_infos = [], []
        for i in range(n_eval_episodes):
            episode_length = 0
            episode_reward = 0
            obs, _ = env.reset()
            state = policy.get_initial_state()
            terminated, truncated, prev_action, prev_reward, info = False, False, None, None, None
            infos_this_episode = []

            while not terminated or truncated:
                action, state, _ = policy.compute_single_action(obs=obs, state=state, prev_action=prev_action,
                                                                prev_reward=prev_reward, info=info)
                obs, reward, terminated, truncated, info = env.step(action)

                prev_action, prev_reward = action, reward
                episode_reward += reward
                episode_length += 1
                infos_this_episode.append(info)
            variables = infos_this_episode[0].keys()
            episode_info = {}
            for v in variables:
                episode_info[v] = {}
            for v in variables:
                for info_dict in infos_this_episode:
                    episode_info[v].update(info_dict[v])
            if env.normalize:
                episode_reward = env.norm.unnormalize_rew(episode_reward)
            episode_rewards.append(episode_reward)
            episode_infos.append(episode_info)
        return episode_rewards, episode_infos

    def on_postprocess_trajectory(
            self,
            *,
            worker: "RolloutWorker",
            episode: Union[Episode, EpisodeV2],
            agent_id: AgentID,
            policy_id: PolicyID,
            policies: Dict[PolicyID, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
            **kwargs,
    ) -> None:
        pass

    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        episode.user_data["reward"] = []
        episode.hist_data["reward"] = []
        episode.user_data["action"] = []
        episode.hist_data["action"] = []
        episode.user_data["growth"] = []
        episode.hist_data["growth"] = []
        if "TWSO" in worker.env.crop_features:
            episode.user_data["TWSO"] = []
            episode.hist_data["TWSO"] = []
        if isinstance(base_env.action_space, gym.spaces.MultiDiscrete):
            episode.user_data["measure"] = []
            episode.hist_data["measure"] = []

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        if "TWSO" in worker.env.crop_features:
            episode.user_data["TWSO"].append(episode.last_info_for()["TWSO"][worker.env.date])
        episode.user_data["reward"].append(episode.last_reward_for())
        episode.user_data["action"].append(next(iter(episode.last_info_for()["action"].values())))
        episode.user_data["growth"].append(episode.last_info_for()["growth"][worker.env.date])
        if isinstance(base_env.action_space, gym.spaces.MultiDiscrete):
            episode.user_data["measure"].append(next(iter(episode.last_info_for()["measure"].values())))

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2],
            env_index: Optional[int] = None,
            **kwargs,
    ):
        episode.custom_metrics["reward_mean"] = np.mean(episode.user_data["reward"])
        episode.custom_metrics["reward_median"] = np.median(episode.user_data["reward"])
        episode.custom_metrics["reward"] = episode.user_data["reward"]
        episode.hist_data["reward"] = episode.user_data["reward"]
        episode.custom_metrics["action"] = episode.user_data["action"]
        episode.hist_data["action"] = episode.user_data["action"]
        episode.custom_metrics["growth"] = episode.user_data["growth"]
        episode.hist_data["growth"] = episode.user_data["growth"]
        if "TWSO" in worker.env.crop_features:
            episode.custom_metrics["TWSO"] = episode.user_data["TWSO"]
            episode.hist_data["TWSO"] = episode.user_data["TWSO"]
        if isinstance(base_env.action_space, gym.spaces.MultiDiscrete):
            episode.custom_metrics["measure"] = episode.user_data["measure"]
            episode.hist_data["measure"] = episode.user_data["measure"]

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # print(result['custom_metrics'])

        policy = algorithm.get_policy()

        writer = SummaryWriter(log_dir=algorithm.logdir)

        reward, fertilizer, result_model = {}, {}, {}

        env_pcse_evaluation = ww_lim(algorithm.config['env_config'])

        print("evaluating environment with learned policy...")
        for year in algorithm.config['env_config']["eval_years"]:
            for test_location in algorithm.config['env_config']["eval_locations"]:
                env_pcse_evaluation.overwrite_year(year)
                env_pcse_evaluation.overwrite_location(test_location)
                env_pcse_evaluation.reset()
                episode_rewards, episode_infos = self.evaluate_rllib_policy(policy=policy, env=env_pcse_evaluation)
                my_key = (year, test_location)
                reward[my_key] = episode_rewards[0].item()
                if env_pcse_evaluation.po_features:
                    episode_infos = eval.get_measure_graphs(episode_infos)
                fertilizer[my_key] = sum(episode_infos[0]['fertilizer'].values())
                writer.add_scalar(f'eval/reward-{my_key}', reward[my_key], result["timesteps_total"])
                writer.add_scalar(f'eval/nitrogen-{my_key}', fertilizer[my_key], result["timesteps_total"])
                result_model[my_key] = episode_infos

        for test_location in algorithm.config['env_config']["eval_locations"]:
            test_keys = [(a, test_location) for a in algorithm.config['env_config']["eval_years"]]
            writer.add_scalar(f'eval/reward-average-test-{test_location}', eval.compute_average(reward, test_keys),
                              result["timesteps_total"])
            writer.add_scalar(f'eval/nitrogen-average-test-{test_location}',
                              eval.compute_average(fertilizer, test_keys), result["timesteps_total"])
            writer.add_scalar(f'eval/reward-median-test-{test_location}', eval.compute_median(reward, test_keys),
                              result["timesteps_total"])
            writer.add_scalar(f'eval/nitrogen-median-test-{test_location}',
                              eval.compute_median(fertilizer, test_keys), result["timesteps_total"])

        if env_pcse_evaluation.pcse_model:
            variables = ['DVS', 'action', 'TWSO', 'reward',
                         'fertilizer', 'val']
            if env_pcse_evaluation.po_features:
                variables.append('measure')
                for p in env_pcse_evaluation.po_features:
                    variables.append(p)
            if env_pcse_evaluation.reward_function == 'ANE': variables.append('moving_ANE')
        else:
            variables = ['action', 'WSO', 'reward', 'TNSOIL', 'val']
            if env_pcse_evaluation.po_features: variables.append('measure')

        if 'measure' in variables:
            variables.remove('measure')
            for variable in env_pcse_evaluation.po_features:
                variable = 'measure_' + variable
                variables += [variable]

        keys_figure = [(a, b) for a in algorithm.config['env_config']["eval_years"]
                       for b in algorithm.config['env_config']["eval_locations"]]
        results_figure = {filter_key: result_model[filter_key] for filter_key in keys_figure}

        for i, variable in enumerate(variables):
            if variable not in results_figure[list(results_figure.keys())[0]][0].keys():
                continue
            plot_individual = False
            if plot_individual:
                fig, ax = plt.subplots()
                eval.plot_variable(results_figure, variable=variable, ax=ax, ylim=eval.get_ylim_dict()[variable])
                writer.add_figure(f'figures/{variable}', fig, result["timesteps_total"])
                plt.close()

            fig, ax = plt.subplots()
            eval.plot_variable(results_figure, variable=variable, ax=ax, ylim=eval.get_ylim_dict()[variable],
                               plot_average=True)
            writer.add_figure(f'figures/avg-{variable}', fig, result["timesteps_total"])
            plt.close()
