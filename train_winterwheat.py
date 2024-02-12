import argparse
import lib_programname
import sys
import os.path
import time
import json

# For comet use
use_comet = True
if use_comet:
    from comet_ml import Experiment
    from comet_ml.integration.gymnasium import CometLogger
import torch.nn as nn

import gymnasium.spaces

from pcse_gym.envs.constraints import ActionConstrainer
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_policy_kwargs, get_model_kwargs
from pcse_gym.utils.eval import EvalCallback, determine_and_log_optimum
from pcse_gym.utils.normalization import VecNormalizePO
import pcse_gym.utils.defaults as defaults

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[0]

if rootdir not in sys.path:
    print(f'insert {os.path.join(rootdir)}')
    sys.path.insert(0, os.path.join(rootdir))

if os.path.join(rootdir, "pcse_gym") not in sys.path:
    sys.path.insert(0, os.path.join(rootdir, "pcse_gym"))


def wrapper_vectorized_env(env_pcse_train, flag_po, normalize=False):
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    if normalize:
        return DummyVecEnv([lambda: env_pcse_train])
    if flag_po:
        return VecNormalizePO(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                              clip_obs=10., clip_reward=50., gamma=1)
    else:
        return VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                            clip_obs=10., clip_reward=50., gamma=1)

def get_json_config(n_steps, crop_features, weather_features, train_years, test_years, train_locations, test_locations,
                    action_space, pcse_model, agent, reward, seed, costs_nitrogen, kwargs):
    return dict(
        n_steps=n_steps,crop_features=crop_features,weather_features=weather_features,
        train_years=train_years,test_years=test_years,train_locations=train_locations,
        test_locations=test_locations,action_space=action_space,pcse_model=pcse_model,agent=agent,reward=reward,
        seed=seed,costs_nitrogen=costs_nitrogen,kwargs=kwargs
    )


def train(log_dir, n_steps,
          crop_features=defaults.get_wofost_default_crop_features(),
          weather_features=defaults.get_default_weather_features(),
          action_features=defaults.get_default_action_features(),
          train_years=defaults.get_default_train_years(),
          test_years=defaults.get_default_test_years(),
          train_locations=defaults.get_default_location(),
          test_locations=defaults.get_default_location(),
          action_space=defaults.get_default_action_space(),
          pcse_model=0, agent='PPO', reward=None,
          seed=0, tag="Exp", costs_nitrogen=10.0,
          **kwargs):
    """
    Train a PPO agent (Stable Baselines3).

    Parameters
    ----------
    log_dir: directory where the (tensorboard) data will be saved
    n_steps: int, number of timesteps the agent spends in the environment
    crop_features: crop features
    weather_features: weather features
    action_features: action features
    train_years: train years
    test_years: test years
    train_locations: train locations (latitude,longitude)
    test_locations: test locations (latitude,longitude)
    action_space: action space
    pcse_model: 0 for LINTUL else WOFOST
    agent: one of {PPO, RPPO, DQN}
    reward: one of {DEF, GRO, or ANE}
    seed: random seed
    tag: tag for tensorboard and friends
    costs_nitrogen: float, penalty for fertilization application

    """
    if agent == 'DQN':
        assert not kwargs.get('args_measure'), f'cannot use {agent} with measure args'

    pcse_model_name = "LINTUL" if not pcse_model else "WOFOST"

    action_limit = kwargs.get('action_limit', 0)
    flag_po = kwargs.get('po_features', [])
    n_budget = kwargs.get('n_budget', 0)
    framework = kwargs.get('framework', 'sb3')
    no_weather = kwargs.get('no_weather', False)
    normalize = kwargs.get('normalize', False)
    mask_binary = kwargs.get('mask_binary', False)
    loc_code = kwargs.get('loc_code', None)
    random_init = kwargs.get('random_init', False)
    cost_measure = kwargs.get('cost_measure', None)
    measure_all =  kwargs.get('measure_all', None)

    if framework == 'sb3':
        from stable_baselines3 import PPO, DQN, A2C
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

        print('Using the StableBaselines3 framework')
        print(f'Train model {pcse_model_name} with {agent} algorithm and seed {seed}. Logdir: {log_dir}')

        if agent == 'PPO':
            hyperparams = {'batch_size': 64, 'n_steps': 2048, 'learning_rate': 0.0002, 'ent_coef': 0.0,
                           'clip_range': 0.3,
                           'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.4,
                           'policy_kwargs': {},
                           }
            if not no_weather:
                hyperparams['policy_kwargs'] = get_policy_kwargs(n_crop_features=len(crop_features),
                                                                 n_weather_features=len(weather_features),
                                                                 n_action_features=len(action_features),
                                                                 n_po_features=len(po_features) if flag_po else 0,
                                                                 mask_binary=mask_binary)
            hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
            hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
            hyperparams['policy_kwargs']['ortho_init'] = False
        if agent == 'RPPO':
            hyperparams = {'batch_size': 64, 'n_steps': 2048, 'learning_rate': 0.0001, 'ent_coef': 0.0,
                           'clip_range': 0.4,
                           'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.4,
                           'policy_kwargs': {},
                           }
            if not no_weather:
                hyperparams['policy_kwargs'] = get_policy_kwargs(n_crop_features=len(crop_features),
                                                                 n_weather_features=len(weather_features),
                                                                 n_action_features=len(action_features),
                                                                 n_po_features=len(po_features) if flag_po else 0,
                                                                 mask_binary=mask_binary)
            hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
            hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
            hyperparams['policy_kwargs']['ortho_init'] = False
        if agent == 'A2C':
            hyperparams = {'n_steps': 2048, 'learning_rate': 0.0002, 'ent_coef': 0.0,
                           'gae_lambda': 0.9, 'vf_coef': 0.4,  # 'rms_prop_eps': 1e-5, 'normalize_advantage': True,
                           'policy_kwargs': {},
                           }
            if not no_weather:
                hyperparams['policy_kwargs'] = get_policy_kwargs(n_crop_features=len(crop_features),
                                                                 n_weather_features=len(weather_features),
                                                                 n_action_features=len(action_features))
            hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
            hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
            hyperparams['policy_kwargs']['ortho_init'] = False
            # hyperparams['policy_kwargs']['optimizer_class'] = RMSpropTFLike
            # hyperparams['policy_kwargs']['optimizer_kwargs'] = dict(eps=0.00001)
        if agent == 'DQN':
            hyperparams = {'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0,
                           'exploration_final_eps': 0.01,
                           'policy_kwargs': get_policy_kwargs(n_crop_features=len(crop_features),
                                                              n_weather_features=len(weather_features),
                                                              n_action_features=len(action_features))
                           }
            hyperparams['policy_kwargs']['net_arch'] = [256, 256]
            hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh

        if use_comet:
            with open(os.path.join(rootdir, 'comet', 'hbja_api_key'), 'r') as f:
                api_key = f.readline()
            comet_log = Experiment(
                api_key=api_key,
                project_name="cropGym_new_wofost",
                workspace="pcse-gym",
                log_code=True,
                log_graph=True,
                auto_metric_logging=True,
                auto_histogram_tensorboard_logging=True
            )
            comet_log.log_code(folder=os.path.join(rootdir, 'pcse_gym'))
            comet_log.log_parameters(hyperparams)

        env_pcse_train = WinterWheat(crop_features=crop_features, action_features=action_features,
                                     weather_features=weather_features,
                                     costs_nitrogen=costs_nitrogen, years=train_years, locations=train_locations,
                                     action_space=action_space, action_multiplier=1.0, seed=seed,
                                     reward=reward, **get_model_kwargs(pcse_model, train_locations,
                                                                       start_type=kwargs.get('start_type', 'sowing')),
                                     **kwargs)

        env_pcse_train = Monitor(env_pcse_train)

        env_pcse_train = ActionConstrainer(env_pcse_train, action_limit=action_limit, n_budget=n_budget)

        if use_comet and comet_log:
            env_pcse_train = CometLogger(env_pcse_train, comet_log)
            comet_log.add_tags(['sb3', agent, seed, loc_code, reward])

        if agent == 'PPO':
            env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po, normalize)
            model = PPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                        tensorboard_log=log_dir)
        elif agent == 'DQN':
            env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po, normalize)
            model = DQN('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                        tensorboard_log=log_dir)
        elif agent == 'A2C':
            env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po, normalize)
            model = A2C('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                        tensorboard_log=log_dir)
        elif agent == 'RPPO':
            from sb3_contrib import RecurrentPPO
            env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po, normalize)
            model = RecurrentPPO('MlpLstmPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                                 tensorboard_log=log_dir)

        compute_baselines = False
        if compute_baselines:
            determine_and_log_optimum(log_dir, env_pcse_train,
                                      train_years=train_years, test_years=test_years,
                                      train_locations=train_locations, test_locations=test_locations,
                                      n_steps=args.nsteps)

        env_pcse_eval = WinterWheat(crop_features=crop_features, action_features=action_features,
                                    weather_features=weather_features,
                                    costs_nitrogen=costs_nitrogen, years=test_years, locations=test_locations,
                                    action_space=action_space, action_multiplier=1.0, reward=reward,
                                    **get_model_kwargs(pcse_model, train_locations, start_type=kwargs.get('start_type', 'sowing')),
                                    **kwargs, seed=seed)
        if action_limit or n_budget > 0:
            env_pcse_eval = ActionConstrainer(env_pcse_eval, action_limit=action_limit, n_budget=n_budget)

        if measure_all:
            cost_measure = 'all'
        tb_log_name = f'{tag}-{loc_code}-nsteps-{n_steps}-{agent}-rew-{reward}-lim-act-{action_limit}-budget-{n_budget}'
        if cost_measure:
            tb_log_name = cost_measure + '-' + tb_log_name
        if no_weather:
            tb_log_name = tb_log_name + '-no_weather'
        if normalize:
            tb_log_name = tb_log_name + '-normalize'
        if mask_binary:
            tb_log_name = tb_log_name + '-masked'
        if random_init:
            tb_log_name = tb_log_name + '-random-init'
        if use_comet:
            comet_log.set_name(f'{tag}-{agent}-{cost_measure}')
            comet_log.add_tag(cost_measure)
        tb_log_name = tb_log_name + '-run'

        # json_config = get_json_config(n_steps, crop_features, weather_features, train_years, test_years,
        #                               train_locations, test_locations, action_space, pcse_model, agent,
        #                               reward, seed, costs_nitrogen, kwargs)
        json_config = locals()

        print(json_config)


        model.learn(total_timesteps=n_steps,
                    callback=EvalCallback(env_eval=env_pcse_eval, test_years=test_years,
                                          train_years=train_years, train_locations=train_locations,
                                          test_locations=test_locations, seed=seed, pcse_model=pcse_model,
                                          comet_experiment=comet_log, **kwargs),
                    tb_log_name=tb_log_name)

        # from stable_baselines3.common.utils import get_latest_run_id
        # latest_run_id = get_latest_run_id(log_dir, tb_log_name)
        # tb_log_name = f'{tb_log_name}_{latest_run_id}'
        #
        # with open(os.path.join(log_dir, tb_log_name, 'config_of_run.json'), 'w') as f:
        #     json.dump(json_config, f)
    elif framework == 'rllib':
        import ray
        from ray import tune
        from ray.tune.registry import register_env
        from pcse_gym.utils.rllib_helpers import (get_algo_config, winterwheat_config_maker, get_algo, ww_lim,
                                                  ww_lim_norm,
                                                  ww_nor, ww_unwrapped_unnormalized, modify_algo_config)

        print('Using the RLlib framework')
        log_dir_ = os.path.join(rootdir, "tensorboard_logs/rllib")
        print(f'Train model {pcse_model_name} with {agent} algorithm and seed {seed}. Logdir: {log_dir_}')
        env_config = winterwheat_config_maker(crop_features=crop_features, action_features=action_features,
                                              weather_features=weather_features,
                                              costs_nitrogen=costs_nitrogen, years=train_years,
                                              eval_years=test_years, locations=train_locations,
                                              eval_locations=test_locations, action_space=action_space,
                                              action_multiplier=1.0, seed=seed,
                                              reward=reward, pcse_model=pcse_model,
                                              **get_model_kwargs(pcse_model, train_locations,
                                                                 start_type=kwargs.get('start_type', 'sowing')),
                                              **kwargs)

        def trial_str_creator(trial):
            prefix = args.agent + "_" + "seed_" + str(seed) + "_" + pcse_model_name
            trialname = prefix + "_" + time.strftime("%Y-%m-%d_%H-%M-%S") + trial.trial_id
            return trialname

        register_env('WinterWheatRay', ww_lim)

        algo_config = get_algo_config(get_algo(agent), env_config, 'WinterWheatRay')
        modify_algo_config(algo_config, agent)

        # algo_config['observation_filter'] = "MeanStdFilter"

        use_comet_ = False
        comet_callback = None
        if use_comet_:
            from ray.air.integrations.comet import CometLoggerCallback
            with open(os.path.join(rootdir, 'comet', 'hbja_api_key'), 'r') as f:
                api_key = f.readline()
            comet_callback = [CometLoggerCallback(
                api_key=api_key,
                project_name='cropGym_1',
                workspace="pcse-gym",
                tags=['rllib', agent, seed, loc_code],
                log_code=True,
                log_graph=True,
                auto_metric_logging=True,
                auto_histogram_tensorboard_logging=True
            )]

        ray.init()

        tune.run(
            "PPO",
            config=algo_config,
            stop={
                # "training_iteration": args.nsteps/1_000
                "timesteps_total": args.nsteps
            },
            local_dir=log_dir_,
            trial_name_creator=trial_str_creator,
            callbacks=comet_callback,
            checkpoint_freq=10,
            checkpoint_at_end=True,
        )
    elif framework == 'ACNO-MDP':
        raise NotImplementedError
        from pcse_gym.agents.BAM_QMDP import BAM_QMDP as ATM
        from pcse_gym.envs.acno_mdp_wrapper import am_env
        from pcse_gym.train_acno_mdp_agents import run_training

        env = WinterWheat(crop_features=crop_features, action_features=action_features,
                          weather_features=weather_features,
                          costs_nitrogen=costs_nitrogen, years=train_years, locations=train_locations,
                          action_space=action_space, action_multiplier=1.0, seed=seed,
                          reward=reward, **get_model_kwargs(pcse_model, train_locations), **kwargs)
        env = am_env(env, 0)

        model = ATM(env, offline_training_steps=0)

        run_training(model, agent, pcse_model_name, n_runs=n_steps / 40, n_eps=20, args=args, do_save=False)
    else:
        raise Exception("Framework choice error!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="Set seed")
    parser.add_argument("-n", "--nsteps", type=int, default=400000, help="Number of steps")
    parser.add_argument("-c", "--costs-nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    parser.add_argument("-e", "--environment", type=int, default=1,
                        help="Crop growth model. 0 for LINTUL-3, 1 for WOFOST")
    parser.add_argument("-a", "--agent", type=str, default="PPO", help="RL agent. PPO, RPPO, GRU,"
                                                                       "IndRNN, DiffNC, PosMLP, ATM or DQN")
    parser.add_argument("-r", "--reward", type=str, default="DEF", help="Reward function. DEF, DEP, GRO, END or ANE")
    parser.add_argument("-b", "--n-budget", type=int, default=0, help="Nitrogen budget. kg/ha")
    parser.add_argument("--action_limit", type=int, default=0, help="Limit fertilization frequency."
                                                                    "Recommended 4 times")
    parser.add_argument("-m", "--measure", action='store_true', help="--measure or --no-measure."
                                                                     "Train an agent in a partially"
                                                                     "observable environment that"
                                                                     "decides when to measure"
                                                                     "certain crop features")
    parser.add_argument("-l", "--location", type=str, default="NL", help="location to train the agent. NL or LT.")
    parser.add_argument("--random-init", action='store_true', dest='random_init')
    parser.add_argument("--no-measure", action='store_false', dest='measure')
    parser.add_argument("--noisy-measure", action='store_true', dest='noisy_measure')
    parser.add_argument("--variable-recovery-rate", action='store_true', dest='vrr')
    parser.add_argument("--no-weather", action='store_true', dest='no_weather')
    parser.add_argument("--random-feature", action='store_true', dest='random_feature')
    parser.add_argument("--mask-binary", action='store_true', dest='obs_mask')
    parser.add_argument("--placeholder-0", action='store_const', const=0.0, dest='placeholder_val')
    parser.add_argument("--normalize", action='store_true', dest='normalize')
    parser.add_argument("--cost-measure", type=str, default='real', dest='cost_measure', help='real, no, or same')
    parser.add_argument("--start-type", type=str, default='sowing', dest='start_type', help='sowing or emergence')
    parser.add_argument("--measure-cost-multiplier", type=int, default=1, dest='m_multiplier', help="multiplier for the measuring cost")
    parser.add_argument("--measure-all", action='store_true', dest='measure_all')
    parser.set_defaults(measure=True, vrr=False, noisy_measure=False, framework='sb3',
                        no_weather=False, random_feature=False, obs_mask=False, placeholder_val=-1.11,
                        normalize=False, random_init=False, m_multiplier=1, measure_all=False)

    args = parser.parse_args()

    # making sure everything is compatible with the user choices
    if not args.measure and args.noisy_measure:
        parser.error("noisy measure should be used with measure")
    if args.agent not in ['PPO', 'A2C', 'RPPO', 'DQN', 'GRU', 'PosMLP', 'S4D', 'IndRNN', 'DiffNC', 'ATM']:
        parser.error("Invalid agent argument. Please choose PPO, A2C, RPPO, GRU, IndRNN, DiffNC, PosMLP, ATM, DQN")
    if args.agent in ['GRU', 'PosMLP', 'S4D', 'IndRNN', 'DiffNC']:
        args.framework = 'rllib'
    elif args.agent in ['ATM']:
        args.framework = 'ACNO-MDP'
    pcse_model_name = "LINTUL" if not args.environment else "WOFOST"

    # directory where the model is saved
    print(rootdir)
    log_dir = os.path.join(rootdir, 'tensorboard_logs', f'{pcse_model_name}_experiments')
    print(f'train for {args.nsteps} steps with costs_nitrogen={args.costs_nitrogen} (seed={args.seed})')

    # define training and testing years
    all_years = [*range(1990, 2022)]
    train_years = [year for year in all_years if year % 2 == 1]
    test_years = [year for year in all_years if year % 2 == 0]

    # define training and testing locations
    if args.location == "NL":
        """The Netherlands"""
        train_locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
        test_locations = [(52, 5.5), (48, 0)]
    elif args.location == "LT":
        """Lithuania"""
        train_locations = [(55.0, 23.5), (55.0, 24.0), (55.5, 23.5)]
        test_locations = [(46.0, 25.0), (55.0, 23.5)]  # Romania
    else:
        parser.error("--location arg should be either LT or NL")

    # define the crop, weather and (maybe) action features used in training
    crop_features = defaults.get_default_crop_features(pcse_env=args.environment, minimal=True)
    weather_features = defaults.get_default_weather_features()
    action_features = defaults.get_default_action_features()

    tag = f'Seed-{args.seed}'

    # define key word arguments
    kwargs = {'args_vrr': args.vrr, 'action_limit': args.action_limit, 'noisy_measure': args.noisy_measure,
              'n_budget': args.n_budget, 'framework': args.framework, 'no_weather': args.no_weather,
              'mask_binary': args.obs_mask, 'placeholder_val': args.placeholder_val, 'normalize': args.normalize,
              'loc_code': args.location, 'cost_measure': args.cost_measure, 'start_type': args.start_type,
              'random_init': args.random_init, 'm_multiplier': args.m_multiplier, 'measure_all': args.measure_all}

    # define MeasureOrNot environment if specified
    if not args.measure:
        action_spaces = gymnasium.spaces.Discrete(7)  # 7 levels of fertilizing
    else:
        if args.environment:
            po_features = ['TAGP', 'LAI', 'NAVAIL', 'NuptakeTotal', 'SM']
            if args.random_feature:
                po_features.append('random')
                crop_features.append('random')
        else:
            po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
        kwargs['po_features'] = po_features
        kwargs['args_measure'] = True
        if not args.noisy_measure:
            m_shape = 2
        else:
            m_shape = 3
        if args.measure_all:
            a_shape = [7] + [m_shape]
        else:
            a_shape = [7] + [m_shape] * len(po_features)
        action_spaces = gymnasium.spaces.MultiDiscrete(a_shape)

    train(log_dir, train_years=train_years, test_years=test_years,
          train_locations=train_locations,
          test_locations=test_locations,
          n_steps=args.nsteps, seed=args.seed,
          tag=tag, costs_nitrogen=args.costs_nitrogen,
          crop_features=crop_features,
          weather_features=weather_features,
          action_features=action_features, action_space=action_spaces,
          pcse_model=args.environment, agent=args.agent,
          reward=args.reward, **kwargs)
