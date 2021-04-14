"""
Infers fine-scale recombination maps from data.

Functions:
    optimize: Builds and optimizes an objective functon.

Run on the command line with pyrho optimzie -h or --help to see
a list of options and their description.
"""
from __future__ import division
import time
import logging
import sys
from itertools import repeat, chain
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np
from pandas import read_hdf
from scipy.optimize import minimize_scalar

from pyrho.haplotype_reader import (genos_to_configs, parse_vcf_to_genos,
                                    parse_seqs_to_genos)
from pyrho.rho_splines import compute_splines
from pyrho.objective_function import RhomapObjective
from pyrho.utility import _single_vec_downsample, downsample, InterruptablePool as Pool


# the following algorithms implement Liu et al. 2010
def _liu_alg_1_step(x_i, x_im1, alpha_i, alpha_im1, L, bpen, obj_func):
    beta = (alpha_im1 - 1) / alpha_i
    si = x_i + beta*(x_i - x_im1)
    si = np.minimum(si, np.log(1e4))
    si = np.maximum(si, np.log(1e-20))
    curr_val, curr_grad = obj_func.value_and_grad(si)
    L_curr = L
    while True:
        y_val = _liu_alg_2(si, curr_grad, bpen, L_curr, obj_func)
        resid = y_val - si
        hl_val = curr_val + curr_grad.dot(resid) + 0.5*L_curr*(resid**2).sum()
        h_val = obj_func.value(y_val)
        if h_val <= hl_val:
            break
        if L_curr/L >= 10:
            L = L_curr
        L_curr += L
        if L_curr > 1000:
            return (x_i, x_i, 0, 0, float('inf'), curr_val)
    alpha = 0.5*(1 + np.sqrt(1 + 4*alpha_i**2))
    return (y_val, x_i, alpha, alpha_i, L_curr, h_val)


def _liu_alg_2(si, curr_grad, bpen, L, obj_func):
    lamb_2 = bpen / L
    v = si - curr_grad / L
    delta_v = obj_func.delta_matrix.dot(v)
    z_star_unconstrained = _rose_alg(delta_v)
    lamb_max = np.abs(z_star_unconstrained).max()
    if lamb_2 >= lamb_max:
        z_star = z_star_unconstrained
    else:
        z_star = _liu_alg_3(v, delta_v, lamb_2, z_star_unconstrained,
                            obj_func.delta_matrix, obj_func.rrt)
    return v - obj_func.delta_matrix.transpose().dot(z_star)


def _rose_alg(u):
    n = len(u) + 1
    s = -np.arange(1, n).dot(u) / n
    z = np.zeros_like(u)
    z[::-1] = np.cumsum(u[::-1]) + s
    z = np.cumsum(z)
    return z


def _liu_alg_3(v, delta_v, lamb_2, z, delta_mat, rrt):
    n = len(v) + 1
    L = 2 - 2*np.cos(np.pi * (n-1) / n)
    error = float('inf')
    g = rrt.dot(z) - delta_v
    prev_error = 0
    restart = True
    max_its = 1000
    its = 0
    while error > 1e-12 and its < 100:
        z = np.minimum(np.maximum(z - g/L, -lamb_2), lamb_2)
        if restart:
            z = _liu_alg_4(v, delta_v, lamb_2, z, delta_mat, rrt)
        g = rrt.dot(z) - delta_v
        error = (lamb_2*np.abs(g).sum() + z.dot(g)) / n
        if prev_error == error and restart:
            restart = False
        prev_error = error
        its += 1
    if its == max_its:
        logging.warning('Liu Algorithm 3 did not converge in 1000 iterations.')
    return z


def _liu_alg_4(v, delta_v, lamb_2, z, delta_mat, rrt):
    g = rrt.dot(z) - delta_v
    boundary_set = (np.abs(z) == lamb_2) * (z*g < 0)
    support_set_interior = np.arange(1, len(z)+1, dtype=int)[boundary_set]
    support_set = np.zeros(len(support_set_interior)+2, dtype=int)
    support_set[1:-1] = support_set_interior
    support_set[-1] = len(z) + 1
    x = np.zeros(len(z) + 1)
    z_conv = np.zeros(z.shape[0] + 1)
    z_conv[0:-1] = z
    for lower, upper in zip(support_set[:-1], support_set[1:]):
        inner_product = v[lower:upper].sum()
        fill_value = inner_product - z_conv[lower-1] + z_conv[upper-1]
        fill_value /= upper - lower
        x[lower:upper] = fill_value
    z_0 = _rose_alg(delta_v - delta_mat.dot(x))
    z_0 = np.minimum(np.maximum(z_0, -lamb_2), lamb_2)
    return z_0


def _window_optimize(args):
    return _window_optimize_helper(*args)


def _window_optimize_helper(genos, lengths, windowsize, table,
                            rho_grid, subtable_sizes, max_size,
                            ploidy, bpen, fast_missing):
    configs, adj_matrix = genos_to_configs(genos, windowsize, ploidy)
    logging.debug('Reticulating splines...')
    splines = compute_splines(configs, table, subtable_sizes,
                              max_size, fast_missing)
    obj_func = RhomapObjective(splines, rho_grid, adj_matrix, lengths, bpen)
    logging.debug('Starting optimization...')
    ml_result = minimize_scalar(obj_func.negative_loglihood_one_rho,
                                bounds=(np.log(1e-20), np.log(4)),
                                method='bounded')
    r_ml = ml_result.x
    logging.debug('ML single rho = %f', np.exp(r_ml))
    start_x = np.full(len(lengths), r_ml)
    start_xm1 = np.array(start_x)
    alpha_i = 1
    alpha_im1 = 0
    L = 1
    i = 1
    error = float('inf')
    prev_val = float('inf')
    inc_count = 0
    while error > 1e-12 and i < 100:
        (start_x,
         start_xm1,
         alpha_i,
         alpha_im1,
         L,
         func_val) = _liu_alg_1_step(start_x, start_xm1, alpha_i,
                                     alpha_im1, L, bpen, obj_func)
        if np.any(np.isnan(start_x)):
            assert not np.any(np.isnan(start_xm1))
            start_x = start_xm1
        start_x = np.maximum(start_x, np.log(1e-20))
        start_x = np.minimum(start_x, np.log(1e4))
        error = np.abs(start_x - start_xm1).max()
        i += 1
        if func_val > prev_val:
            inc_count += 1
            if inc_count > 10:
                inc_count = 0
                L *= 2
                logging.warning('Increasing Regularization, because '
                                'likelihood increased too frequently')
    logging.debug('Chunk complete')
    return np.exp(start_x)


def _stitch(rhos_list, overlap):
    if len(rhos_list) == 1:
        return rhos_list[0]
    if overlap == 0:
        return np.array(list(chain.from_iterable(rhos_list)))
    logging.info('Stitching results with overlap = %d', overlap)
    first_window = rhos_list[0][:-overlap].flatten().tolist()
    bulk = chain.from_iterable([r[overlap:-overlap] for r in rhos_list[1:-1]])
    last_window = rhos_list[-1][overlap:].flatten().tolist()
    return np.array(first_window + list(bulk) + last_window)


def optimize(genos, ploidy, positions, table, rho_list, subtable_sizes,
             max_size,
             metawindow, overlap, windowsize, block_penalty, fast_missing,
             pool=None):
    """
    Infers a fine-scale recombination map.

    Builds an objective function based on data and then optimizes that
    objective function using proximal gradient descent, returns the
    resulting recombination map.

    Args:
        genos: A num_loci by num_samples array containing the genotypes.
        ploidy: The ploidy of the data (1 for phased data, 2 for unphased).
        positions: An array containing the physical locations of the SNPs
            in genos.
        table: The values from the pandas.DataFrame containing a lookup table
            as computed with make_table. If fast_missing is true, then this
            should contain the values from lookup tables for sample sizes
            2...n concatenated.
        rho_list: The rho values corresponding to the columns of table
        subtable_sizes: A numpy array of integers containing the (cummulative)
           sizes for each of the subtables making up table (if fast_missing
           is true).  If fast_missing is false, this is ignored.
        max_size: The number of haplotypes present in the dataset.
        metawindow: The minimum size in number of SNPs into which to chunk the
            data for parallelization.
        overlap: The amount of overlap in number of SNPs to use when stitching
            together recombination maps inferred for each metawindow.
        windowsize: Ignore pairs of SNPs that do not fit into a window of size
            windowsize.
        block_penalty: The L1 regularization penalty used in the fused-LASSO
            objective function.  Higher penalties result in smoother
            recombination maps.
        fast_missing: A boolean which determines whether to use approximations
            to speed up inference.  If fast_mssing is true, then table and
            subtable_sizes need to be adjusted as described above.
        pool: A Multiprocessing.Pool object to perform the parallelization. If
            None, no parallelization is performed.

    Returns:
        A numpy array containing the inferred recombination rate between each
        pair of adjacent SNPs in genos.
    """
    lens = np.diff(positions)
    hap_chunks = []
    lengths_list = []
    k = 0
    non_overlap = (metawindow - 2*overlap - 1)
    logging.debug('Splitting sequence into metawindows.')
    logging.debug('There are %d individuals, and %d loci',
                  genos.shape[1], genos.shape[0])
    while True:
        if len(genos) - (k + 1)*non_overlap < overlap:
            end = len(genos)
        else:
            end = min([k*non_overlap + metawindow, len(genos)])
        hap_chunks.append(genos[k*non_overlap: end])
        lengths_list.append(lens[k*non_overlap:  end-1])
        if end == len(genos):
            break
        k += 1
    logging.debug('\tDone! There are %d metawindows.', k + 1)
    optimization_args = izip(hap_chunks, lengths_list, repeat(windowsize),
                             repeat(table), repeat(rho_list),
                             repeat(subtable_sizes), repeat(max_size),
                             repeat(ploidy),
                             repeat(block_penalty), repeat(fast_missing))
    if pool is None:
        results = list(map(_window_optimize, optimization_args))
    else:
        results = pool.map(_window_optimize, optimization_args)
    final_rhos = _stitch(results, overlap)
    assert len(final_rhos) == len(lens)
    return final_rhos


def _args(super_parser):
    parser = super_parser.add_parser(
        'optimize',
        description='Compute the penalized maximum composite likelihood '
                    'recombination map from a set of genotypes or haplotypes.',
        usage='pyrho optimze <options>'
    )
    required = parser.add_argument_group('required arguments')
    parser.add_argument('-f', '--fastafile', type=str, required=False,
                        help='FASTA format file containing the haplotypes.')
    parser.add_argument('-o', '--outfile', type=str, required=False,
                        help='Destination for printing output. '
                             'Defaults to stdout.')
    parser.add_argument('--vcffile', type=str, required=False,
                        help='VCF format file containing the haplotypes.')
    parser.add_argument('-bpen', '--blockpenalty', type=float, required=False,
                        default=40.0, help='Penalty to ensure smoothness. '
                                           'Larger penalties result in '
                                           'smoother recombination maps.')
    parser.add_argument('-s', '--sitesfile', type=str, required=False,
                        help='LDHat format sites file.')
    parser.add_argument('-l', '--locsfile', type=str, required=False,
                        help='LDHat format locs file.')
    parser.add_argument('-w', '--windowsize', type=int, required=False,
                        default=50, help='Maximum distance to consider '
                                         'between pairs of SNPs.')
    parser.add_argument('--numthreads', type=int, required=False, default=1,
                        help='Number of threads to run in parallel.')
    parser.add_argument('--metawindow', type=int, required=False, default=4001,
                        help='Minimum number of SNPs to include in each '
                             'independent, parallelizable chunk.')
    parser.add_argument('--overlap', type=int, required=False, default=100,
                        help='Amount to trim off each side of a metawindow.')
    parser.add_argument('--ploidy', type=int, required=False, default=1,
                        help='Ploidy of the data.  Currently only support '
                             '1 for haplotypes, 2 for genotypes.')
    required.add_argument('--tablefile', required=True, type=str,
                          help='File containing a pyrho lookup table.')
    parser.add_argument('--vcfpass', required=False, type=str,
                        help='Require that the FILTER column of VCF '
                             'matches vcfpass to inclide SNP in analysis. '
                             'Deault is to include all SNPs.')
    parser.add_argument('--fast_missing',
                        dest='fast_missing',
                        action='store_true',
                        help='Cache some additional likelihoods, and '
                             'for each pair of loci throw away any '
                             'individuals that are missing at exactly '
                             'one locus.  This should be substantially '
                             'faster for datasets with a high degree of '
                             'missingness at a minimal loss of accuracy, '
                             'but it is an untested feature.')
    return parser


def _main(args):
    if (
            not args.fastafile and not (args.sitesfile and args.locsfile)
            and not args.vcffile
    ):
        raise IOError('Must input data using either fastafile, vcffile, '
                      'or sitesfile and locsfile.')
    if args.blockpenalty < 0:
        raise IOError('blockpenalty must be positive.')
    if args.metawindow <= 2*args.overlap:
        raise IOError('metawindow must be more than doulble overlap.')
    if args.windowsize > args.metawindow:
        raise IOError('windowsize cannot be greater than metawindow.')
    if args.fastafile or args.sitesfile:
        if (
                args.fastafile and args.sitesfile
                or args.fastafile and args.locsfile
                or args.vcffile
        ):
            raise IOError('Cannot specify more than one inpute file.')
        if args.sitesfile and not args.locsfile:
            raise IOError('Must specify a locsfile when using a sitesfile.')
        seq_file = args.fastafile if args.fastafile else args.sitesfile
        genos, positions = parse_seqs_to_genos(seq_file, args.ploidy,
                                               args.locsfile)
    if args.vcffile:
        genos, positions = parse_vcf_to_genos(
            args.vcffile, args.ploidy, args.vcfpass
        )
    max_size = genos.shape[1] * args.ploidy
    logging.info('Loading lookup table...')
    table = read_hdf(args.tablefile, 'ldtable')
    max_rho = table.columns[-1]
    table.columns *= 100./max_rho
    logging.info('\tDone!')
    table_size = sum(map(int, table.index.values[0].split()))
    if table_size < max_size:
        raise IOError('Lookup table was constructed for {} haploids, '
                      'but there are as many as {} haploids in the sample. '
                      'Either build a larger lookup table or subsample '
                      'the data.'.format(table_size, max_size))
    if table_size > max_size:
        table = downsample(table, max_size)
    table_list = [table.values]
    subtable_sizes = [table.shape[0]]
    if args.fast_missing:
        logging.info('Generating partial lookup tables')
        curr_size = max_size - 1
        while curr_size > 1:
            table_list.append(
                _single_vec_downsample(table_list[-1], curr_size+1)
            )
            subtable_sizes.append(table_list[-1].shape[0])
            curr_size -= 1
    subtable_sizes = np.cumsum(subtable_sizes[::-1])
    full_table = np.concatenate(table_list[::-1])
    pool = None
    if args.numthreads > 1:
        pool = Pool(args.numthreads, maxtasksperchild=5)
    logging.info('Beginning optimization.')
    start_time = time.time()
    result = optimize(genos, args.ploidy, positions, full_table,
                      table.columns, subtable_sizes, max_size,
                      args.metawindow, args.overlap, args.windowsize,
                      args.blockpenalty, args.fast_missing, pool)
    opt_time = time.time() - start_time
    logging.info('Time for running optimization = %f', opt_time)
    outfile = args.outfile if args.outfile else sys.stdout
    with open(outfile, 'w') as ofile:
        for start, end, rho in zip(positions[:-1], positions[1:], result):
            line = [str(start), str(end), str(rho * max_rho / 100.)]
            ofile.write('\t'.join(line) + '\n')
