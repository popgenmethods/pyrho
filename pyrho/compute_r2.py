"""
Computes theoretical quantiles or mean of the distribution of  r^2.

Run on the command line with pyrho compute_r2 -h or --help to see
a list of options and their description.
"""
from __future__ import division
import sys

import numpy as np
from pandas import read_hdf, DataFrame
from numba import njit

from pyrho.utility import log_mult_coef, downsample


@njit('float64(int64[:, :])', cache=True)
def _log_comb_factor(config):
    num_unique = np.unique(config).shape[0]
    swaps = 2 ** (num_unique - 1)
    if num_unique == 2:
        if config[0, 0] != config[1, 1] or config[0, 1] != config[1, 0]:
            swaps *= 2
    log_mult = log_mult_coef(config.flatten())
    return log_mult + np.log(swaps)


@njit('float64[:, :](float64[:], boolean, float64, int64, float64[:, :])',
      cache=True)
def _compute_statistics(quantiles, compute_mean, maf, sample_size, table):
    probs = np.zeros(table.shape)
    r_sq = np.zeros(table.shape[0])
    idx = 0
    for i in range(1, sample_size//2 + 1):
        for j in range(1, i + 1):
            for k in range(j, -1, -1):
                config = np.array(((sample_size - i - j + k, i - k),
                                   (j - k, k)))
                freqs = config / sample_size
                rho_probs = np.exp(table[idx, :] + _log_comb_factor(config))
                probs[idx, :] = rho_probs
                l_freqs = freqs.sum(axis=1)
                r_freqs = freqs.sum(axis=0)
                if np.all(l_freqs > maf) and np.all(r_freqs > maf):
                    r_sq[idx] = ((freqs[1, 1] - l_freqs[1]*r_freqs[1]) ** 2
                                 / np.prod(l_freqs) / np.prod(r_freqs))
                else:
                    r_sq[idx] = -1
                idx += 1
    if not np.all(probs.sum(axis=0) < 1.0 + 1e-10):
        raise ValueError('Probabilities sum to greater than 1.0')
    r_sq = r_sq[r_sq > -1]
    probs = probs[r_sq > -1, :]
    probs /= probs.sum(axis=0)
    statistics = np.zeros((table.shape[1], compute_mean + len(quantiles)))
    if compute_mean:
        statistics[:, 0] = probs.transpose().dot(r_sq)
    if quantiles is not None:
        sort_args = np.argsort(r_sq)
        r_sq = r_sq[sort_args]
        for rho in range(table.shape[1]):
            cdf = np.cumsum(probs[:, rho][sort_args])
            these_quantiles = r_sq[np.searchsorted(cdf, quantiles)]
            statistics[rho, compute_mean:] = these_quantiles
    return statistics


def _args(super_parser):
    parser = super_parser.add_parser(
        'compute_r2',
        description='Compute theoretical quantiles or mean of the '
                    'distribution of r^2, using a lookupt table '
                    'generated with the pyrho make_table command.',
        usage='pyrho compute_r2 <options>'
    )
    required = parser.add_argument_group('required arguments')
    parser.add_argument('--MAFcut', type=float, required=False, default=0.0,
                        help='Only look at pairs of loci where MAF > MAFcut '
                             'at both loci.')
    parser.add_argument('--quantiles', required=False, type=str, default='',
                        help='Quantiles of r^2 distribution to compute.')
    parser.add_argument('--compute_mean', action='store_true',
                        help='Compute the expected value of r^2.')
    required.add_argument('--tablefile', required=True, type=str,
                          help='Lookup table file.')
    required.add_argument('--samplesize', type=int, required=True,
                          help='Number of haplotypes for which to compute '
                               'statistics.  Must be less than or equal to '
                               'the sample size in tablefile.')
    parser.add_argument('-o', '--outfile', type=str, required=False,
                        default='', help='Destination for printing '
                                         'output.  Defaults to stdout.')
    return parser


def _main(args):
    table = DataFrame(read_hdf(args.tablefile, 'ldtable'))
    first_config = table.index.values[0].split()
    table_sample_size = sum(map(int, first_config))
    if args.MAFcut < 0 or args.MAFcut >= 0.5:
        raise IOError('MAFcut must be between 0 and 0.5')
    if table_sample_size < args.samplesize:
        raise IOError('Cannot compute r^2 statistics for a sample size of {}, '
                      'which is larger than your lookup table, which was '
                      'constructed for a max size of '
                      '{}.'.format(args.samplesize, table_sample_size))
    if table_sample_size > args.samplesize:
        table = downsample(table, args.samplesize)
    rho_grid = np.array(table.columns)
    labels = ['Rho']
    quants = None
    if args.compute_mean:
        labels.append('mean')
    if args.quantiles != '':
        labels += args.quantiles.split(',')
        quants = np.array(list(map(float, args.quantiles.split(','))))
    statistics = _compute_statistics(quants, args.compute_mean, args.MAFcut,
                                     args.samplesize, table.values)
    to_print = ['\t'.join(labels)]
    for rho, rho_stat in zip(rho_grid, statistics):
        to_print.append('\t'.join(map(str, [rho] + rho_stat.tolist())))
    if args.outfile:
        with open(args.outfile, 'w') as ofh:
            ofh.write('\n'.join(to_print))
    else:
        sys.stdout.write('\n'.join(to_print))
        sys.stdout.write('\n')
