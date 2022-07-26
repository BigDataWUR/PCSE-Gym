import os
import sys
import argparse
import lib_programname

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[2]
if rootdir not in sys.path:
    print(f'insert {os.path.join(rootdir)}')
    sys.path.insert(0, os.path.join(rootdir))

from helper import train

if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, help="Set seed")
    args = parser.parse_args()

    log_dir = os.path.join(rootdir, 'notebooks', 'nitrogen-winterwheat', 'tensorboard_logs', 'ExperimentsLong')
    costs_nitrogen = 10.0
    print(f'train with costs_nitrogen={costs_nitrogen}')
    train_years = [2014, 2015, 2016]
    test_years = [2018, 2019, 2020]
    #determine_and_log_optimum(log_dir, costs_nitrogen=costs_nitrogen, train_years=train_years, test_years=test_years)

    train(log_dir, n_steps=400000, seed=args.seed, tag=f'AllFeatures-seed-{args.seed}', costs_nitrogen=costs_nitrogen)
