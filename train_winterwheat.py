from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger

import os.path

import gymnasium.spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import RecurrentPPO  # MaskablePPO
# from sb3_contrib.common.envs import InvalidActionEnvDiscrete
# from sb3_contrib.common.maskable.utils import get_action_masks
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.wrappers import ActionMasker

import argparse
import lib_programname
import sys

from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import *
from pcse_gym.utils.eval import EvalCallback, determine_and_log_optimum

# from pcse_gym.envs.constraints import ActionLimiter

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[0]

if rootdir not in sys.path:
    print(f'insert {os.path.join(rootdir)}')
    sys.path.insert(0, os.path.join(rootdir))

if os.path.join(rootdir, "pcse_gym") not in sys.path:
    sys.path.insert(0, os.path.join(rootdir, "pcse_gym"))


def train(log_dir, n_steps,
          crop_features=get_wofost_default_crop_features(),
          weather_features=get_default_weather_features(),
          action_features=get_default_action_features(),
          train_years=get_default_train_years(),
          test_years=get_default_test_years(),
          all_years=get_default_years(),
          train_locations=get_default_location(),
          test_locations=get_default_location(),
          all_locations=get_default_location(),
          action_space=get_default_action_space(),
          pcse_model=0, agent=PPO, reward=None,
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
    all_years: union of train_years and test_years
    train_locations: train locations (latitude,longitude)
    test_locations: test locations (latitude,longitude)
    all_locations: union of train_locations and test_locations
    seed: random seed
    tag: tag for tensorboard and friends
    costs_nitrogen: float, penalty for fertilization application

    """

    if agent == 'DQN':
        assert not kwargs.get('args_measure'), f'cannot use {agent} with measure args'

    pcse_model_name = "WOFOST" if not pcse_model else "LINTUL"

    print(f'Train model {pcse_model_name} with {agent} algorithm and seed {seed}. Logdir: {log_dir}')
    if agent == 'PPO' or 'RPPO':
        hyperparams = {'batch_size': 64, 'n_steps': 2048, 'learning_rate': 0.0003, 'ent_coef': 0.0, 'clip_range': 0.3,
                       'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.5,
                       'policy_kwargs': get_policy_kwargs(crop_features=crop_features,
                                                          weather_features=weather_features,
                                                          action_features=action_features)}
        hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
        hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
        hyperparams['policy_kwargs']['ortho_init'] = False
    if agent == 'DQN':
        hyperparams = {'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.01,
                       'policy_kwargs': get_policy_kwargs(crop_features=crop_features,
                                                          weather_features=weather_features,
                                                          action_features=action_features)
                       }
        hyperparams['policy_kwargs']['net_arch'] = [256, 256]
        hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh

    # For comet use
    use_comet = True

    if use_comet:
        with open(os.path.join(rootdir, 'comet', 'hbja_api_key'), 'r') as f:
            api_key = f.readline()
        comet_log = Experiment(
            api_key=api_key,
            project_name="experimental_cropgym",
            workspace="pcse-gym",
            log_graph=True,
            auto_metric_logging=True,
            auto_histogram_tensorboard_logging=True
        )

        comet_log.log_parameters(hyperparams)
        comet_log.log_asset_folder(os.path.join(rootdir, 'pcse_gym', 'envs'), log_file_name=True, recursive=True)
        comet_log.log_asset(os.path.join(rootdir, 'pcse_gym', 'utils', 'eval'), file_name='eval.py')

    env_pcse_train = WinterWheat(crop_features=crop_features, action_features=action_features,
                                 weather_features=weather_features,
                                 costs_nitrogen=costs_nitrogen, years=train_years, locations=train_locations,
                                 all_years=all_years, all_locations=all_locations,
                                 action_space=action_space, action_multiplier=1.0, seed=seed,
                                 reward=reward, **get_pcse_model(pcse_model), **kwargs)
    # env_pcse_train = ActionLimiter(env_pcse_train, action_limit=4)

    # env_pcse_train = ActionMasker(env_pcse_train, mask_fertilization_actions)

    env_pcse_train = Monitor(env_pcse_train)

    if comet_log:
        env_pcse_train = CometLogger(env_pcse_train, comet_log)

    match agent:
        case 'PPO':
            env_pcse_train = VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                                          clip_obs=10., clip_reward=50., gamma=1)
            model = PPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams, tensorboard_log=log_dir)
        case 'DQN':
            env_pcse_train = VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                                          clip_obs=10000., clip_reward=5000., gamma=1)
            model = DQN('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                        tensorboard_log=log_dir)
        case 'RPPO':
            env_pcse_train = VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                                          clip_obs=10., clip_reward=50., gamma=1)
            model = RecurrentPPO('MlpLstmPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                                 tensorboard_log=log_dir)
        case _:
            env_pcse_train = VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                                          clip_obs=10., clip_reward=50., gamma=1)
            model = PPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams, tensorboard_log=log_dir)

    debug = False
    if debug:
        print(model.policy)
        actions, values, log_probs = model.policy(get_test_tensor(crop_features, action_features, weather_features))
        print(f'actions: {actions} values: {values} log_probs: {log_probs} probs: {np.exp(log_probs.detach().numpy())}')
        return

    env_pcse_eval = WinterWheat(crop_features=crop_features, action_features=action_features,
                                weather_features=weather_features,
                                costs_nitrogen=costs_nitrogen, years=test_years, locations=test_locations,
                                all_years=all_years, all_locations=all_locations,
                                action_space=action_space, action_multiplier=1.0, reward=reward,
                                **get_pcse_model(pcse_model), **kwargs, seed=seed)
    # env_pcse_eval = ActionLimiter(env_pcse_eval, action_limit=4)

    tb_log_name = f'{tag}-{pcse_model_name}-Ncosts-{costs_nitrogen}-run'

    model.learn(total_timesteps=n_steps,
                callback=EvalCallback(env_eval=env_pcse_eval, test_years=test_years,
                                      train_years=train_years, train_locations=train_locations,
                                      test_locations=test_locations, seed=seed, pcse_model=pcse_model,
                                      **kwargs),
                tb_log_name=tb_log_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="Set seed")
    parser.add_argument("-n", "--nsteps", type=int, default=400000, help="Number of steps")
    parser.add_argument("-c", "--costs_nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    parser.add_argument("-e", "--environment", type=int, default=1,
                        help="Crop growth model. 0 for LINTUL-3, 1 for WOFOST")
    parser.add_argument("-a", "--agent", type=str, default="PPO", help="RL agent. PPO, RPPO, or DQN.")
    parser.add_argument("-r", "--reward", type=str, default="DEF", help="Reward function. DEF, GRO, or ANE")
    parser.add_argument("-m", "--measure", type=bool, default=True, help="Train an agent in a partially observable"
                                                                         "environment that decides when to measure"
                                                                         "certain crop features")
    args = parser.parse_args()

    print(rootdir)
    log_dir = os.path.join(rootdir, 'tensorboard_logs', 'WOFOST_experiments')
    print(f'train for {args.nsteps} steps with costs_nitrogen={args.costs_nitrogen} (seed={args.seed})')

    all_years = [*range(1990, 2022)]
    train_years = [year for year in all_years if year % 2 == 1]
    test_years = [year for year in all_years if year % 2 == 0]

    train_locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
    test_locations = [(52, 5.5), (48, 0)]
    all_locations = list(set(train_locations + test_locations))

    compute_baselines = False
    if compute_baselines:
        determine_and_log_optimum(log_dir, costs_nitrogen=args.costs_nitrogen,
                                  train_years=train_years, test_years=test_years,
                                  train_locations=train_locations, test_locations=test_locations,
                                  n_steps=args.nsteps)
    if args.environment:
        # see https://github.com/ajwdewit/pcse/tree/develop_WOFOST_v8_1/pcse
        crop_features = ["DVS", "TAGP", "LAI", "NuptakeTotal", "NAVAIL", "SM"]
    else:
        # see https://github.com/ajwdewit/pcse/blob/master/pcse/crop/lintul3.py
        crop_features = ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN", "TNSOIL", "TRAIN", "TRANRF", "WSO"]
    weather_features = ["IRRAD", "TMIN", "RAIN"]
    action_features = []  # alternative: "cumulative_nitrogen"
    tag = f'Seed-{args.seed}'

    if not args.measure:
        action_spaces = gymnasium.spaces.Discrete(7)
    else:
        if args.environment:
            po_features = ['TAGP', 'LAI', 'NAVAIL', 'NuptakeTotal', 'SM']
        else:
            po_features = ['TGROWTH', 'LAI', 'TNSOIL', 'NUPTT', 'TRAIN']
        kwargs = {'po_features': po_features, 'args_measure': True}
        a_shape = [7] + [2] * len(po_features)
        action_spaces = gymnasium.spaces.MultiDiscrete(a_shape)

    train(log_dir, train_years=train_years, test_years=test_years,
          train_locations=train_locations,
          test_locations=test_locations,
          n_steps=args.nsteps, seed=args.seed,
          tag=tag, all_years=all_years, all_locations=all_locations,
          costs_nitrogen=args.costs_nitrogen,
          crop_features=crop_features,
          weather_features=weather_features,
          action_features=action_features, action_space=action_spaces,
          pcse_model=args.environment, agent=args.agent,
          reward=args.reward, **kwargs)
