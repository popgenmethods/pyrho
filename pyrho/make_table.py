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


def _args(super_parser):
    parser = super_parser.add_parser(
        'make_table',
        description='Precompute a lookup table using ldpop.',
        usage='pyrho make_table <options>'
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument('-n', '--samplesize', type=int, required=True,
                          help='Maximum number of haplotypes in your sample.')
    required.add_argument('-th', '--theta', type=float, required=True,
                          help='Twice the population-scaled mutation rate.')
    parser.add_argument('-t', '--epochtimes', type=str, required=False,
                        default='', help='Comma delimitted list of epoch '
                                         'breakpoints in coalescent units.')
    parser.add_argument('-p', '--popsizes', type=str, required=False,
                        default='1.0', help='Comma delimitted list of epoch '
                                            'breakpoints')
    parser.add_argument('-N', '--moran_pop_size', type=int, required=False,
                        default=None, help='Number of particles to consider '
                                           'if using --approx')
    parser.add_argument('--numthreads', type=int, required=False, default=1,
                        help='Number of threads to run in parallel')
    parser.add_argument('--approx', action='store_true',
                        help='Use the Moran approximation to compute the '
                             'haplotype lookup table')
    required.add_argument('-o', '--outfile', required=True, type=str,
                          help='Name of file to store computed '
                               'lookup table in')
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
    return parser


def _main(args):
    times = []
    if args.epochtimes:
        times = list(map(float, args.epochtimes.split(',')))
    pop_sizes = list(map(float, args.popsizes.split(',')))
    if len(pop_sizes) != len(times)+1:
        raise IOError('Number of population sizes must match '
                      'number of epochs.')
    max_size = args.samplesize
    num_particles = max_size
    if args.moran_pop_size:
        if not args.approx:
            raise IOError('Cannot use moran_pop_size when computing an exact '
                          'lookup table.  Turn off --aprox flag.')
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

    if num_particles > max_size:
        logging.info('Downsampling')
        table = downsample(table, max_size)
        logging.info('\t...complete')
    table.to_hdf(args.outfile, 'ldtable', mode='w')
