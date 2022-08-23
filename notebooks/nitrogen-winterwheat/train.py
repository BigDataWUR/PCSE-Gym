import os
import sys
import argparse
import lib_programname

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[2]
if rootdir not in sys.path:
    print(f'insert {os.path.join(rootdir)}')
    sys.path.insert(0, os.path.join(rootdir))

from helper import train, determine_and_log_optimum

if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="Set seed")
    parser.add_argument("-n", "--nsteps", type=int, default=400000, help="Number of steps")
    parser.add_argument("-c", "--costs_nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    args = parser.parse_args()

    log_dir = os.path.join(rootdir, 'notebooks', 'nitrogen-winterwheat', 'tensorboard_logs', 'Experiments-ManyYears')
    print(f'train for {args.nsteps} steps with costs_nitrogen={args.costs_nitrogen} (seed={args.seed})')
    all_years = [*range(1990, 2022)]
    train_years = [year for year in all_years if year % 2 == 1]
    test_years = [year for year in all_years if year % 2 == 0]

    determine_and_log_optimum(log_dir, costs_nitrogen=args.costs_nitrogen, train_years=train_years, test_years=test_years, n_steps=args.nsteps)
    train(log_dir, train_years=train_years, test_years=test_years, n_steps=args.nsteps, seed=args.seed, tag=f'AllFeatures-seed-{args.seed}', costs_nitrogen=args.costs_nitrogen)
