import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.monitor import Monitor
from torch import nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Union
import datetime
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize_scalar
import torch.nn.functional as F
import torch
import os
from wrapper import ReferenceEnv
from wrapper import get_default_crop_features, get_default_weather_features, get_default_train_years, get_default_test_years


def get_prob(model, obs):
    observation = np.array(obs)
    observation = observation.reshape((-1,) + model.observation_space.shape)
    observation = torch.as_tensor(observation).to(model.device)
    with torch.no_grad():
        features = model.policy.extract_features(observation)
        latent_pi, latent_vf = model.policy.mlp_extractor(features)
        mean_actions = model.policy.action_net(latent_pi)
        probabilities = F.softmax(mean_actions, dim=-1).cpu().numpy()
    return np.argmax(probabilities), np.amax(probabilities)


def evaluate_policy(
        policy,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        amount = 1
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param policy: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param amount: Multiplier for action
    :return: a list of reward per episode
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    action_space = env.action_space
    if isinstance(action_space, list):
        action_space = action_space[0]

    if isinstance(policy, base_class.BaseAlgorithm):
        assert(policy.action_space == action_space)

    if isinstance(action_space, gym.spaces.Discrete) and not isinstance(policy, base_class.BaseAlgorithm):
        print('Warning!')

    episode_rewards, episode_infos = [], []
    for i in range(n_eval_episodes):
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        if isinstance(policy, base_class.BaseAlgorithm):
            sync_envs_normalization(policy.get_env(), env)
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        year = env.get_attr("date")[0].year
        fert_dates = [datetime.date(year, 2, 24), datetime.date(year, 3, 26), datetime.date(year, 4, 29)]
        action = [amount * 0]
        infos_this_episode = []
        prob = None

        while not done:
            if policy == 'start-dump' and (episode_length==0):
                action = [amount * 1]
            if isinstance(policy, base_class.BaseAlgorithm):
                action, state = policy.predict(obs, state=state, deterministic=deterministic)
                _, prob = get_prob(policy, obs)

            obs, reward, done, info = env.step(action)
            if prob:
                action_date = list(info[0]['action'].keys())[0]
                info[0]['prob'] = {action_date: prob}

            action = [amount * 0]
            if policy in ['standard-practice', 'standard-practise']:
                date = env.get_attr("date")[0]
                for fert_date in fert_dates:
                    if date > fert_date and date <= fert_date + datetime.timedelta(7):
                        action = [amount *3]
            if policy == 'no-nitrogen':
                action = [0]
            episode_reward += reward
            episode_length += 1
            infos_this_episode.append(info[0])
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        episode_rewards.append(episode_reward)
        episode_infos.append(episode_info)
    return episode_rewards, episode_infos


class FindOptimum():
    def __init__(self, env, train_years=None):
        self.train_years = train_years
        self.env = env
        if train_years is None:
            self.train_years = [env.get_attr("date")[0].year]

    def start_dump(self, x):
        def def_value(): return 0
        self.current_rewards = defaultdict(def_value)
        for train_year in self.train_years:
            self.env.env_method('overwrite_year', train_year)
            self.env.reset()
            done = False
            infos_this_episode = []
            total_reward = 0.0
            while not done:
                action = [0.0]
                if len(infos_this_episode)==0:
                    action = [x * 1.0]
                info_this_episode, reward, done, _ = self.env.step(action)
                total_reward = total_reward + reward
                infos_this_episode.append(info_this_episode)
            self.current_rewards[self.env.get_attr("date")[0].year] = total_reward
        returnvalue = 0
        for year, reward in self.current_rewards.items():
            returnvalue = returnvalue - reward
        return returnvalue

    def optimize_start_dump(self):
        res = minimize_scalar(self.start_dump, bounds=(0, 100.0), method='bounded')
        print(f'optimum found for {self.train_years} at {res.x} {-1.0*res.fun}')
        for year, reward in self.current_rewards.items():
            print(f'- {year} {reward}')
        return res.x


def get_ylim_dict():
    def def_value():
        return None
    ylim = defaultdict(def_value)
    ylim['WSO'] = [0, 1000]
    return ylim


def get_titles():
    def def_value(): return ("", "")
    return_dict = defaultdict(def_value)
    return_dict["DVS"] = ("Development stage", "-")
    return_dict["TGROWTH"] = ("Total biomass (above and below ground)", "g/m2")
    return_dict["LAI"] = ("Leaf area Index", "-")
    return_dict["NUPTT"] = ("Total nitrogen uptake", "gN/m2")
    return_dict["TRAN"] = ("Transpiration", "mm/day")
    return_dict["TIRRIG"] = ("Total irrigation", "mm")
    return_dict["TNSOIL"] = ("Total soil inorganic nitrogen", "gN/m2")
    return_dict["TRAIN"] = ("Total rainfall", "mm")
    return_dict["TRANRF"] = ("Transpiration reduction factor", "-")
    return_dict["TRUNOF"] = ("Total runoff", "mm")
    return_dict["TAGBM"] = ("Total aboveground biomass", "g/m2")
    return_dict["TTRAN"] = ("Total transpiration", "mm")
    return_dict["WC"] = ("Soil water content", "m3/m3")
    return_dict["WLVD"] = ("Weight dead leaves", "g/m2")
    return_dict["WLVG"] = ("Weight green leaves", "g/m2")
    return_dict["WRT"] = ("Weight roots", "g/m2")
    return_dict["WSO"] = ("Weight storage organs", "g/m2")
    return_dict["WST"] = ("Weight stems", "g/m2")
    return return_dict


def plot_variable(results_dict, variable='reward', cumulative_variables=None, ax=None, ylim=None, put_legend=True):
    if cumulative_variables is None:
        cumulative_variables = ['fertilizer', 'reward']

    titles = get_titles()
    xmax=0
    for label, results in results_dict.items():
        x, y = zip(*results[0][variable].items())
        x = ([i.timetuple().tm_yday for i in x])
        if variable in cumulative_variables: y = np.cumsum(y)
        if max(x) > xmax: xmax = max(x)
        ax.step(x, y, label=label, where='post')

    for x in range(0, xmax, 7):
        ax.axvline(x=x, color='lightgrey', zorder=1)
    ax.axhline(y=0, color='lightgrey', zorder=1)
    ax.margins(x=0)
    name, unit = titles[variable]
    ax.set_title(f"{variable} - {name}")
    ax.set_ylabel(f"[{unit}]")
    if ylim != None:
        ax.set_ylim(ylim)
    if put_legend:
        ax.legend()
    return ax


def compute_average(results_dict: dict, filter_list=None):
    if filter_list is None:
        filter_list = list(results_dict.keys())
    filtered_results = [results_dict[f] for f in filter_list]
    return sum(filtered_results) / len(filtered_results)


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, train_years=get_default_train_years(), test_years=get_default_test_years(), n_eval_episodes=1, eval_freq=1000):
        super(EvalCallback, self).__init__()
        self.test_years = test_years
        self.train_years = train_years
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        def def_value(): return 0
        self.histogram_training_years = defaultdict(def_value)


    def _on_step(self):
        """
        This method will be called by the model.

        :return: (bool)
        """

        train_year = self.model.get_env().get_attr("date")[0].year
        self.histogram_training_years[train_year] = self.histogram_training_years[train_year] + 1

        if self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            tensorboard_logdir = self.logger.dir
            model_path = os.path.join(tensorboard_logdir, f'model-{self.n_calls}')
            self.model.save(model_path)
            stats_path = os.path.join(tensorboard_logdir, f'env-{self.n_calls}.pkl')
            self.model.get_env().save(stats_path)
            episode_rewards, episode_infos = evaluate_policy(policy=self.model, env=self.model.get_env())
            variables = 'action', 'WSO', 'reward', 'TNSOIL'
            cumulative = ['action', 'reward']
            for i, variable in enumerate(variables):
                n_timepoints = len(episode_infos[0][variable])
                n_episodes = len(episode_infos)
                episode_results = np.empty((n_episodes, n_timepoints))
                episode_summary = np.empty(n_episodes)
                for e in range(n_episodes):
                    _, y = zip(*episode_infos[e][variable].items())
                    if variable in cumulative: y = np.cumsum(y)
                    episode_results[e, :] = y
                    episode_summary[e] = y[-1]
                variable_mean = np.mean(episode_summary, axis=0)
                self.logger.record(f'train/{variable}', variable_mean)

            fig, ax = plt.subplots()
            ax.bar(range(len(self.histogram_training_years)), list(self.histogram_training_years.values()), align='center')
            ax.set_xticks(range(len(self.histogram_training_years)), minor=False)
            ax.set_xticklabels(list(self.histogram_training_years.keys()), fontdict=None, minor=False, rotation=90)
            self.logger.record(f'figures/training-years', Figure(fig, close=True))

            costs_nitrogen = list(self.model.get_env().get_attr('costs_nitrogen'))[0]
            action_multiplier = list(self.model.get_env().get_attr('action_multiplier'))[0]
            action_space = self.model.get_env().get_attr('action_space')
            crop_features = self.model.get_env().get_attr('crop_features')[0]
            weather_features = self.model.get_env().get_attr('weather_features')[0]

            reward, fertilizer, result_model = {}, {}, {}
            for year in self.test_years + self.train_years:
                env_pcse_evaluation = ReferenceEnv(crop_features, weather_features, costs_nitrogen=costs_nitrogen,
                                                   years=year, action_space=action_space, action_multiplier=action_multiplier)
                env_pcse_evaluation = VecNormalize(DummyVecEnv([lambda: env_pcse_evaluation]), norm_obs=True, norm_reward=False,
                                              clip_obs=10., gamma=1)
                sync_envs_normalization(self.model.get_env(), env_pcse_evaluation)
                env_pcse_evaluation.training, env_pcse_evaluation.norm_reward = False, False
                episode_rewards, episode_infos = evaluate_policy(policy=self.model, env=env_pcse_evaluation)
                reward[year] = episode_rewards[0].item()
                fertilizer[year] = sum(episode_infos[0]['fertilizer'].values())
                self.logger.record(f'eval/reward-{year}', reward[year])
                self.logger.record(f'eval/nitrogen-{year}', fertilizer[year])
                result_model[year] = episode_infos
                del env_pcse_evaluation

            self.logger.record(f'eval/reward-average-test', compute_average(reward, self.test_years))
            self.logger.record(f'eval/nitrogen-average-test', compute_average(fertilizer, self.test_years))
            self.logger.record(f'eval/reward-average-train', compute_average(reward, self.train_years))
            self.logger.record(f'eval/nitrogen-average-train', compute_average(fertilizer, self.train_years))

            for i, variable in enumerate(variables):
                fig, ax = plt.subplots()
                plot_variable(result_model, variable=variable, ax=ax, ylim=get_ylim_dict()[variable])
                self.logger.record(f'figures/{variable}', Figure(fig, close=True))
                plt.close()
            self.logger.dump(step=self.num_timesteps)

        return True

def determine_and_log_optimum(log_dir, costs_nitrogen=10.0, train_years=get_default_train_years(), test_years=get_default_test_years(), n_steps=250000):
    print(f'find optimum for {train_years}')
    env_train = ReferenceEnv(costs_nitrogen=costs_nitrogen, years=train_years)
    env_train = VecNormalize(DummyVecEnv([lambda: env_train]), norm_obs=True, norm_reward=False, clip_obs=10., gamma=1)
    optimizer_train = FindOptimum(env_train, train_years)
    optimum_train = optimizer_train.optimize_start_dump()
    optimum_train_path_tb = os.path.join(log_dir, f"Optimum-Ncosts-{costs_nitrogen}-train")
    optimum_train_writer = SummaryWriter(log_dir=optimum_train_path_tb)
    optimum_test_path_tb = os.path.join(log_dir, f"Optimum-Ncosts-{costs_nitrogen}-test")
    optimum_test_writer = SummaryWriter(log_dir=optimum_test_path_tb)

    reward_train, fertilizer_train = {}, {}
    reward_test, fertilizer_test = {}, {}
    for year in train_years + test_years:
        env_test = ReferenceEnv(costs_nitrogen=costs_nitrogen, years=year)
        env_test = VecNormalize(DummyVecEnv([lambda: env_test]), norm_obs=True, norm_reward=False, clip_obs=10., gamma=1)
        optimum_train_rewards, optimum_train_results = evaluate_policy('start-dump', env_test, amount=optimum_train)
        reward_train[year] = optimum_train_rewards[0].item()
        fertilizer_train[year] = sum(optimum_train_results[0]['action'].values())
        print(f'optimum-train: {year} {fertilizer_train[year]} {reward_train[year]}')

        print(f'find optimum for year {year}')
        optimizer_test = FindOptimum(env_test)
        optimum_test = optimizer_test.optimize_start_dump()
        optimum_test_rewards, optimum_test_results = evaluate_policy('start-dump', env_test, amount=optimum_test)
        reward_test[year] = optimum_test_rewards[0].item()
        fertilizer_test[year] = sum(optimum_test_results[0]['action'].values())

        for step in [0, n_steps]:
            optimum_train_writer.add_scalar(f'eval/reward-{year}', reward_train[year], step)
            optimum_train_writer.add_scalar(f'eval/nitrogen-{year}', fertilizer_train[year], step)
            optimum_test_writer.add_scalar(f'eval/reward-{year}', reward_test[year], step)
            optimum_test_writer.add_scalar(f'eval/nitrogen-{year}', fertilizer_test[year], step)
        optimum_train_writer.flush()
        optimum_test_writer.flush()

    for step in [0, n_steps]:
        optimum_train_writer.add_scalar(f'eval/reward-average-test', compute_average(reward_train, test_years), step)
        optimum_train_writer.add_scalar(f'eval/nitrogen-average-test', compute_average(fertilizer_train, test_years), step)
        optimum_train_writer.add_scalar(f'eval/reward-average-train', compute_average(reward_train, train_years), step)
        optimum_train_writer.add_scalar(f'eval/nitrogen-average-train', compute_average(fertilizer_train, train_years), step)

    optimum_train_writer.flush()
    optimum_test_writer.flush()

def determine_and_log_standard_practise(log_dir, costs_nitrogen=10.0,
                                        years=get_default_train_years() + get_default_test_years(),
                                        n_steps=250000):
    path_tb = os.path.join(log_dir, f"Standard-Practise-Ncosts-{costs_nitrogen}")
    writer = SummaryWriter(log_dir=path_tb)
    for year in years:
        env_test = ReferenceEnv(costs_nitrogen=costs_nitrogen, years=year)
        env_test = VecNormalize(DummyVecEnv([lambda: env_test]), norm_obs=True, norm_reward=False, clip_obs=10., gamma=1)
        rewards, results = evaluate_policy('standard-practise', env_test, amount=2)
        reward = rewards[0].item()
        fertilizer = sum(results[0]['fertilizer'].values())
        print(f'standard-practise: {year} {fertilizer} {reward}')
        for step in [0, n_steps]:
            writer.add_scalar(f'eval/reward-{year}', reward, step)
            writer.add_scalar(f'eval/nitrogen-{year}', fertilizer, step)


def train(log_dir, n_steps,
          crop_features=get_default_crop_features(),
          weather_features=get_default_weather_features(),
          train_years=get_default_train_years(),
          test_years=get_default_test_years(),
          seed=0, tag="Exp", costs_nitrogen=10.0):
    """
    Train a PPO agent

    Parameters
    ----------
    log_dir: directory where the (tensorboard) data will be saved
    n_steps: int, number of timesteps the agent spends in the environment
    crop_features: crop features
    weather_features: weather features
    train_years: train years
    test_years: test years
    seed: random seed
    tag: tag for tensorboard and friends
    costs_nitrogen: float, penalty for fertilization application

    """

    print(f'Train model with seed {seed}')
    hyperparams = {'batch_size': 64,
                'n_steps': 2048,
                   'learning_rate': 0.0003,
                   'ent_coef': 0.0,
                   'clip_range': 0.3,
                   'n_epochs': 10,
                   'gae_lambda': 0.95,
                   'max_grad_norm': 0.5,
                   'vf_coef': 0.5,
                   'policy_kwargs': dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                         activation_fn=nn.Tanh,
                                         ortho_init=False)
                  }

    env_pcse_train = ReferenceEnv(crop_features, weather_features, costs_nitrogen=costs_nitrogen, years=train_years,
                                  action_space = gym.spaces.Discrete(3), action_multiplier=2.0)
    env_pcse_train = Monitor(env_pcse_train)
    env_pcse_train = VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=False,
                                              clip_obs=10., gamma=1)

    model = PPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams, tensorboard_log=log_dir)
    model.learn(total_timesteps=n_steps, callback=EvalCallback(test_years=test_years, train_years=train_years), tb_log_name = f'{tag}-Ncosts-{costs_nitrogen}-run')
