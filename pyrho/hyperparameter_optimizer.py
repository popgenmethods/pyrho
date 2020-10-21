"""
Simulates data to determine good optimization hyperparameters.

Uses msprime to simulate data with recombination maps drawn
from the HapMap recombination map, and then infers recombination
maps on the simulated data. Writes various measures of accuracy
(correlation in several forms and scales; L2 norms) to a file.

Run on the command line with pyrho hyperparam -h or --help to
see a lit of options and their description.
"""
from __future__ import division
import logging
import sys
from pkg_resources import resource_stream
from itertools import repeat
from functools import partial
try:
    from itertools import izip as zip
except ImportError:
    pass

import numpy as np
import msprime
from scipy.stats import spearmanr, pearsonr
from pandas import read_hdf
from numba import njit

from pyrho.utility import InterruptablePool as Pool
from pyrho.optimizer import optimize
from pyrho.size_reader import read_msmc, read_smcpp, decimate_sizes


def _simulate_data(sim_args):
    prng = np.random.RandomState()
    msp_args, reco_maps = sim_args
    rmap = reco_maps[prng.choice(len(reco_maps))]
    pop_config, mu, demo, ploidy = msp_args
    reco_map = msprime.RecombinationMap(rmap[0], [r / 40000. for r in rmap[1]])
    label = np.zeros(1000000)
    for start, end, rate in zip(rmap[0][:-1], rmap[0][1:], rmap[1][:-1]):
        label[start:end] = rate
    tree_sequence = msprime.simulate(population_configurations=pop_config,
                                     recombination_map=reco_map,
                                     mutation_rate=mu,
                                     demographic_events=demo)
    haps = np.copy(tree_sequence.genotype_matrix())
    haps = np.sum([haps[:, k::ploidy] for k in range(ploidy)], axis=0)
    mut_pos = [variant.site.position for variant in tree_sequence.variants()]
    return haps, mut_pos, label


def _call_optimize(dataset, metawindow, windowsize, table, ploidy, bpen,
                   overlap, max_rho):
    logging.info('Windowsize = %d, Block Penalty = %f', windowsize, bpen)
    haps, positions, _ = dataset
    result = optimize(haps, ploidy, positions, table, metawindow,
                      overlap, windowsize, bpen, pool=None) * max_rho / 100.
    logging.info('Done optimizing')
    return result


def _window_average(args):
    return _window_average_helper(*args)


@njit('float64[:](float64[:], int64)', cache=True)
def _window_average_helper(array, rate):
    to_return = np.zeros(array.shape[0]//rate)
    for i in range(array.shape[0]):
        to_return[i//rate] += array[i] / rate
    return to_return


def _compute_correlations(vec1, vec2):
    pearson = pearsonr(vec1, vec2)[0]
    log_pearson = pearsonr(np.log(vec1), np.log(vec2))[0]
    spearman = spearmanr(vec1, vec2)[0]
    return [pearson, log_pearson, spearman]


def _score(estimates, positions, labels, pool):
    new_estimates = []
    new_labels = []
    for lab, these_positions, est in zip(labels, positions, estimates):
        e_copy = np.zeros_like(lab)
        first = int(these_positions[0])
        last = int(these_positions[-1])
        for idx, pos in enumerate(zip(these_positions[0:-1],
                                      these_positions[1:])):
            interval_start, interval_end = map(int, pos)
            e_copy[interval_start:interval_end] = est[idx]
        new_estimates.append(e_copy[first:last])
        new_labels.append(lab[first:last])

    estimates_1bp = np.hstack(new_estimates)
    labels_1bp = np.hstack(new_labels) / 40000.
    corr_1bp = pool.apply_async(_compute_correlations,
                                (estimates_1bp, labels_1bp))
    estimates_10kb = pool.map(_window_average,
                              zip(new_estimates, repeat(10000)))
    labels_10kb = pool.map(_window_average,
                           zip(new_labels, repeat(10000)))
    estimates_100kb = pool.map(_window_average,
                               zip(estimates_10kb, repeat(10)))
    labels_100kb = pool.map(_window_average,
                            zip(labels_10kb, repeat(10)))
    estimates_10kb = np.hstack(estimates_10kb)
    labels_10kb = np.hstack(labels_10kb)
    estimates_100kb = np.hstack(estimates_100kb)
    labels_100kb = np.hstack(labels_100kb)

    l2_norm = np.sqrt(((estimates_1bp - labels_1bp)**2).sum())
    log_l2_norm = np.sqrt(((np.log(estimates_1bp)
                            - np.log(labels_1bp))**2).sum())

    to_return = (corr_1bp.get()
                 + _compute_correlations(estimates_10kb, labels_10kb)
                 + _compute_correlations(estimates_100kb, labels_100kb)
                 + [l2_norm, log_l2_norm])
    return to_return


def _load_hapmap():
    reco_maps = []
    for bline in resource_stream('pyrho', 'data/pyrho_hapmap_maps.txt'):
        positions, rates = zip(*[p.split(',') for p in bline.decode().split()])
        positions = list(map(int, map(float, positions)))
        rates = list(map(float, rates))
        assert np.all(np.array(rates)[:-1] > 0)
        reco_maps.append((positions, rates))
    return reco_maps


def _args(super_parser):
    parser = super_parser.add_parser(
        'hyperparam',
        description='Perform a simulation study to do hyperparameter '
                    'optimization.',
        usage='pyrho hyperparam <options>'
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument('-o', '--outfile', type=str, required=True,
                          help='Destination for printing output.')
    parser.add_argument('--numthreads', type=int, required=False, default=1,
                        help='Number of threads to run in parallel.')
    parser.add_argument('--metawindow', type=int, required=False, default=4001,
                        help='Minimum number of SNPs to include in each '
                             'independent, parallelizable chunk.')
    parser.add_argument('--overlap', type=int, required=False, default=100,
                        help='Amount to trim off each side of a metawindow.')
    parser.add_argument('--ploidy', type=int, required=False, default=1,
                        help='Ploidy of the data. Currently only support '
                             '1 for haplotypes, 2 for genotypes.')
    required.add_argument('--tablefile', required=True, type=str,
                          help='File containing a pyrho lookup table.')
    required.add_argument('-n', '--samplesize', type=int, required=True,
                          help='Maximum number of haplotypes in your sample.')
    required.add_argument('-m', '--mu', type=float, required=True,
                          help='The per-generation mutation rate.')
    parser.add_argument('-t', '--epochtimes', type=str, required=False,
                        default='', help='Comma delimitted list of epoch '
                                         'breakpoints in generations.')
    parser.add_argument('--msmc_file', type=str, required=False,
                        default='', help='MSMC output file to specify the '
                                         'size history.')
    parser.add_argument('--smcpp_file', type=str, required=False,
                        default='', help='smc++ csv file to specify the size '
                                         'history.')
    parser.add_argument('-p', '--popsizes', type=str, required=False,
                        default='', help='Comma delimitted list of epoch '
                                         'population sizes.')
    parser.add_argument('--num_sims', type=int, required=False, default=100,
                        help='Number of 1Mb regions to simulate.')
    parser.add_argument('-bpen', '--blockpenalty', type=str, required=False,
                        default='15,20,25,30,35,40,45,50',
                        help='Comma delimited list of block penalties to '
                             'try.')
    parser.add_argument('-w', '--windowsize', type=str, required=False,
                        default='30,40,50,60,70,80,90',
                        help='Comma delimited list of window sizes to try.')
    parser.add_argument('--decimate_rel_tol', required=False, type=float,
                        default=0.0, help='Relative tolerance when decimating '
                                          'size history.')
    parser.add_argument('--decimate_anc_size', required=False, type=float,
                        default=None, help='Most ancestral size when '
                                           'decimating size history.')
    return parser


def _main(args):
    table = read_hdf(args.tablefile, 'ldtable')
    table_size = sum(map(int, table.index.values[0].split()))
    if table_size < args.samplesize:
        raise IOError('Lookup table was constructed for {} haploids, '
                      'but --samplesize is {} haploids.  Either build '
                      'a larger lookup table or simulate fewer '
                      'individuals.'.format(table_size, args.samplesize))
    max_rho = table.columns[-1]
    table.columns *= 100. / max_rho
    block_penalties = list(map(float, args.blockpenalty.split(',')))
    window_sizes = list(map(float, args.windowsize.split(',')))
    logging.info('Searching over Windowsizes %s, and Block Penalties %s',
                 window_sizes, block_penalties)
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
        pop_sizes = list(map(float, args.popsizes.split(',')))
        times = []
        if args.epochtimes:
            times = list(map(float, args.epochtimes.split(',')))
    if len(pop_sizes) != len(times) + 1:
        raise IOError('Number of population sizes must '
                      'match number of epochs.')
    pop_sizes, times = decimate_sizes(pop_sizes,
                                      times,
                                      args.decimate_rel_tol,
                                      args.decimate_anc_size)

    pop_config = [
        msprime.PopulationConfiguration(sample_size=args.samplesize,
                                        initial_size=pop_sizes[0])]
    demography = []
    if times:
        for pop_size, time in zip(pop_sizes[1:], times):
            demography.append(
                msprime.PopulationParametersChange(time=time * 2,
                                                   initial_size=pop_size,
                                                   population_id=0))
    reco_maps = _load_hapmap()
    pool = Pool(args.numthreads, maxtasksperchild=100)
    logging.info('Simulating data...')
    simulation_args = [((pop_config, args.mu, demography, args.ploidy),
                        reco_maps) for k in range(args.num_sims)]
    test_set = list(pool.imap(_simulate_data, simulation_args, chunksize=10))
    logging.info('\tdone simulating')
    scores = {}
    for block_penalty in block_penalties:
        for window_size in window_sizes:
            estimates = list(pool.imap(partial(_call_optimize,
                                               metawindow=args.metawindow,
                                               windowsize=window_size,
                                               table=table,
                                               ploidy=args.ploidy,
                                               bpen=block_penalty,
                                               overlap=args.overlap,
                                               max_rho=max_rho),
                                       test_set,
                                       chunksize=10))
            scores[(block_penalty,
                    window_size)] = _score(estimates,
                                           [ts[1] for ts in test_set],
                                           [ts[2] for ts in test_set],
                                           pool)
    ofile = open(args.outfile, 'w') if args.outfile else sys.stdout
    ofile.write('\t'.join(['Block_Penalty',
                           'Window_Size',
                           'Pearson_Corr_1bp',
                           'Pearson_Corr_10kb',
                           'Pearson_Corr_100kb',
                           'Log_Pearson_Corr_1bp',
                           'Log_Pearson_Corr_10kb',
                           'Log_Pearson_Corr_100kb',
                           'Spearman_Corr_1bp',
                           'Spearman_Corr_10kb',
                           'Spearman_Corr_100kb',
                           'L2',
                           'Log_L2']) + '\n')
    for block_penalty, window_size in sorted(scores):
        line = ([block_penalty, window_size]
                + scores[block_penalty, window_size])
        ofile.write('\t'.join(map(str, line)) + '\n')
    if args.outfile:
        ofile.close()
