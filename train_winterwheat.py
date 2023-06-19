from stable_baselines3 import PPO#, DQN
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import argparse
import lib_programname
import sys

from pcse_gym.utils.defaults import *
from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import *
from pcse_gym.utils.eval import EvalCallback, determine_and_log_optimum


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
          train_locations=get_default_location(),
          test_locations=get_default_location(),
          pcse_model=0, agent=PPO,
          seed=0, tag="Exp", costs_nitrogen=10.0):
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
    seed: random seed
    tag: tag for tensorboard and friends
    costs_nitrogen: float, penalty for fertilization application

    """
    pcse_model_name = lambda x: "WOFOST" if x == 1 else "LINTUL"
    print(f'Train model {pcse_model_name(pcse_model)} with seed {seed}. Logdir: {log_dir}')
    hyperparams = {'batch_size': 64, 'n_steps': 2048, 'learning_rate': 0.0003, 'ent_coef': 0.0, 'clip_range': 0.3,
                   'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.5,
                   'policy_kwargs': get_policy_kwargs(crop_features=crop_features, weather_features=weather_features,
                                                      action_features=action_features)}
    hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
    hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
    hyperparams['policy_kwargs']['ortho_init'] = False

    env_pcse_train = WinterWheat(crop_features=crop_features, action_features=action_features,
                                  weather_features=weather_features,
                                  costs_nitrogen=costs_nitrogen, years=train_years, locations=train_locations,
                                  action_space=gym.spaces.Discrete(3), action_multiplier=2.0,
                                 **get_pcse_model(pcse_model))
    env_pcse_train = Monitor(env_pcse_train)
    env_pcse_train = VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                                  clip_obs=10., clip_reward=50., gamma=1)
    model = PPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams, tensorboard_log=log_dir)

    debug = False
    if debug:
        print(model.policy)
        actions, values, log_probs = model.policy(get_test_tensor(crop_features, action_features, weather_features))
        print(f'actions: {actions} values: {values} log_probs: {log_probs} probs: {np.exp(log_probs.detach().numpy())}')
        return


    tb_log_name = f'{tag}-{pcse_model_name(pcse_model)}-Ncosts-{costs_nitrogen}-run'

    model.learn(total_timesteps=n_steps, callback=EvalCallback(test_years=test_years, train_years=train_years,
                train_locations=train_locations, test_locations=test_locations, seed=seed, pcse_model=pcse_model),
                tb_log_name=tb_log_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="Set seed")
    parser.add_argument("-n", "--nsteps", type=int, default=400000, help="Number of steps")
    parser.add_argument("-c", "--costs_nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    parser.add_argument("-e", "--environment", type=int, default=1,
                        help="Crop growth model. 0 for LINTUL-3, 1 for WOFOST")
    parser.add_argument("-a", "--agent", type=str, default="DQN", help="RL agent. PPO or DQN.")
    args = parser.parse_args()

    print(rootdir)
    log_dir = os.path.join(rootdir, 'tensorboard_logs', 'WOFOST_experiments')
    print(f'train for {args.nsteps} steps with costs_nitrogen={args.costs_nitrogen} (seed={args.seed})')
    all_years = [*range(1990, 2022)]
    train_years = [year for year in all_years if year % 2 == 1]
    test_years = [year for year in all_years if year % 2 == 0]

    train_locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
    test_locations = [(52, 5.5), (48, 0)]

    compute_baselines = False
    if compute_baselines:
        determine_and_log_optimum(log_dir, costs_nitrogen=args.costs_nitrogen,
                                  train_years=train_years, test_years=test_years,
                                  train_locations=train_locations, test_locations=test_locations,
                                  n_steps=args.nsteps)
    if args.environment:
        # see https://github.com/ajwdewit/pcse/tree/develop_WOFOST_v8_1/pcse
        crop_features = ["DVS", "TAGP", "LAI", "RNuptake", "TRA", "NAVAIL", "SM", "RFTRA", "TWSO"]
    else:
        # see https://github.com/ajwdewit/pcse/blob/master/pcse/crop/lintul3.py
        crop_features = ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN", "TNSOIL", "TRAIN", "TRANRF", "WSO"]
    weather_features = ["IRRAD", "TMIN", "RAIN"]
    action_features = []  # alternative: "cumulative_nitrogen"
    tag = f'Seed-{args.seed}'

    train(log_dir, train_years=train_years, test_years=test_years,
          train_locations=train_locations,
          test_locations=test_locations,
          n_steps=args.nsteps, seed=args.seed,
          tag=tag,
          costs_nitrogen=args.costs_nitrogen,
          crop_features=crop_features,
          weather_features=weather_features,
          action_features=action_features,
          pcse_model=args.environment, agent=args.agent
          )