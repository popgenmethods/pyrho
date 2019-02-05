"""Constructs parsers for the command line interface."""
import logging
from argparse import ArgumentParser
from pyrho import VERSION
from pyrho.make_table import _main as make_table
from pyrho.make_table import _args as make_table_args
from pyrho.optimizer import _main as optimize
from pyrho.optimizer import _args as optimize_args
from pyrho.hyperparameter_optimizer import _main as hyperparam
from pyrho.hyperparameter_optimizer import _args as hyperparam_args
from pyrho.compute_r2 import _main as compute_r2
from pyrho.compute_r2 import _args as compute_r2_args

COMMANDS = {
    'make_table': {'cmd': make_table, 'parser': make_table_args},
    'optimize': {'cmd': optimize, 'parser': optimize_args},
    'hyperparam': {'cmd': hyperparam, 'parser': hyperparam_args},
    'compute_r2': {'cmd': compute_r2, 'parser': compute_r2_args}
}


def main():
    """
    Takes command line input and calls appropriate pyrho command.

    The available commands are:
        make_table: Build lookup tables to use in other commands.
        hyperparam: Perform simulation studies to determine good
            hyperparameter settings for pyrho optimize.
        optimize: Infer fine-scale recombination maps from sequence data.
        compute_r2: Compute quantiles or mean of the theoretical
            distribution of r^2.
    Calling pyrho <command> --help will show the available options for each
    subcommand.
    """
    parser = ArgumentParser(
        description="""
                    pyrho v%s quickly infers fine-scale recombination
                    maps using composite likelihoods and gradient-based
                    optimization.
                    """ % VERSION,
        usage='pyrho <command> <options>'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    for cmd in COMMANDS:
        cmd_parser = COMMANDS[cmd]['parser'](subparsers)
        cmd_parser.add_argument(
            '--logfile', required=False, type=str, default='',
            help='File to store information about the pyrho run. To print to '
                 'stdout use "."  Defaults to no logging.'
        )
        cmd_parser.add_argument(
            '-v', '--verbosity', required=False, type=int, default=30,
            help='Amount of information to print to logfile on a 0-50 scale.'
        )

    args = parser.parse_args()
    try:
        func = COMMANDS[args.command]['cmd']
    except KeyError:
        parser.print_help()
        exit()
    if args.logfile == ".":
        logging.basicConfig(level=50 - args.verbosity)
    elif args.logfile:
        logging.basicConfig(filename=args.log, level=50 - args.verbosity)
    func(args)


if __name__ == '__main__':
    main()
