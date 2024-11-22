"""
Builds lookup tables using ldpop to use in other pyrho commands.

This is a thin wrapper for ldpop that computes a two-locus
likelihood lookup table and stores the resulting table in hdf format.

Run on the command line with pyrho make_table -h or --help to see
a list of options and their description.
"""
from __future__ import division
import logging

from ldpop import LookupTable

from pyrho.utility import downsample
from pyrho.size_reader import read_msmc, read_smcpp, decimate_sizes


def _args(super_parser):
    parser = super_parser.add_parser(
        'make_table',
        description='Precompute a lookup table using ldpop.',
        usage='pyrho make_table <options>'
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument('-n', '--samplesize', type=int, required=True,
                          help='Maximum number of haplotypes in your sample.')
    required.add_argument('-m', '--mu', type=float, required=True,
                          help='The per-generation mutation rate.')
    parser.add_argument('-t', '--epochtimes', type=str, required=False,
                        default='', help='Comma-delimited list of epoch '
                                         'breakpoints in generations.')
    parser.add_argument('--msmc_file', type=str, required=False,
                        default='', help='MSMC output file to specify the '
                                         'size history.')
    parser.add_argument('--smcpp_file', type=str, required=False,
                        default='', help='smc++ csv file to specify the size '
                                         'history.')
    parser.add_argument('-p', '--popsizes', type=str, required=False,
                        default='', help='Comma-delimited list population '
                                         'sizes.')
    parser.add_argument('-N', '--moran_pop_size', type=int, required=False,
                        default=None, help='Number of particles to consider '
                                           'if using --approx.')
    parser.add_argument('-T', '--theta', type=float, required=False,
                        default=0.0005, help='theta genetic diversity '
                                             'parameter [%(default)s]')
    parser.add_argument('--numthreads', type=int, required=False, default=1,
                        help='Number of threads to run in parallel '
                             '[%(default)s].')
    parser.add_argument('--approx', action='store_true',
                        help='Use the Moran approximation to compute the '
                             'haplotype lookup table.')
    required.add_argument('-o', '--outfile', required=True, type=str,
                          help='Name of file to store computed '
                               'lookup table in.')
    parser.add_argument('-S', '--store_stationary', required=False, type=str,
                        default=None, help='Name of file to save stationary '
                                           'distributions -- useful for '
                                           'computing many lookup tables '
                                           'sequentially.')
    parser.add_argument('-L', '--load_stationary', required=False, type=str,
                        default=None, help='Name of file to load stationary '
                                           'distributions -- useful for '
                                           'computing many lookup tables '
                                           'tables sequentially.')
    parser.add_argument('--decimate_rel_tol', required=False, type=float,
                        default=0.0, help='Relative tolerance when decimating '
                                          'size history [%(default)s].')
    parser.add_argument('--decimate_anc_size', required=False, type=float,
                        default=None, help='Most ancestral size when '
                                           'decimating size history.')
    return parser


def _main(args):
    n_ref = args.theta / (4. * args.mu)
    if args.msmc_file:
        if args.smcpp_file or args.epochtimes or args.popsizes:
            raise IOError('Can only specify one of msmc_file, smcpp_file, or '
                          'popsizes')
        pop_sizes, times = read_msmc(args.msmc_file, args.mu)
    elif args.smcpp_file:
        if args.msmc_file or args.epochtimes or args.popsizes:
            raise IOError('Can only specify one of msmc_file, smcpp_file, or '
                          'popsizes')
        pop_sizes, times = read_smcpp(args.smcpp_file)
    else:
        if not args.popsizes:
            raise IOError('If not using msmc_file or smcpp_file, you must '
                          'provide at least one population size using the '
                          'popsizes argument')
        times = []
        if args.epochtimes:
            times = list(map(float, args.epochtimes.split(',')))
        pop_sizes = list(map(float, args.popsizes.split(',')))

    times = [t / (2. * n_ref) for t in times]
    pop_sizes = [p / n_ref for p in pop_sizes]

    if len(pop_sizes) != len(times)+1:
        raise IOError('Number of population sizes must match '
                      'number of epochs.')
    if not all([t1 - t0 > 0 for t1, t0 in zip(times[1:], times[:-1])]):
        raise IOError('Demographies must be specified backward '
                      'in time, with the breakpoints being '
                      'strictly increasing.')
    if not all([p > 0 for p in pop_sizes]):
        raise IOError('Population sizes must be positive.')
    pop_sizes, times = decimate_sizes(pop_sizes,
                                      times,
                                      args.decimate_rel_tol,
                                      args.decimate_anc_size)
    logging.info('Size history to be used when computing lookup table is\n'
                 + 'Scaled Size\tScaled Left Time\tScaled Right Time\n'
                 + '\n'.join([str(p) + '\t' + str(t1) + '\t' + str(t2)
                              for p, t1, t2 in zip(pop_sizes,
                                                   [0] + times,
                                                   times + [float('inf')])]))
    max_size = args.samplesize
    num_particles = max_size
    if args.moran_pop_size:
        if not args.approx:
            raise IOError('Cannot use moran_pop_size when computing an exact '
                          'lookup table.  Turn off --approx flag.')
        if max_size > args.moran_pop_size:
            raise IOError('moran_pop_size must be at least as large as the '
                          'desired sample size.')
        num_particles = args.moran_pop_size

    rho_grid = [i * .1 for i in range(100)] + list(range(10, 101))
    logging.info('Beginning Lookup Table.  This may take a while')
    table = LookupTable(num_particles, args.theta, rho_grid, pop_sizes,
                        times, not args.approx, args.numthreads,
                        store_stationary=args.store_stationary,
                        load_stationary=args.load_stationary).table
    logging.info('\t...complete')
    table.columns /= 4. * n_ref
    if num_particles > max_size:
        logging.info('Downsampling')
        table = downsample(table, max_size)
        logging.info('\t...complete')
    table.to_hdf(args.outfile, key='ldtable', mode='w')
