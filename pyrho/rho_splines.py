"""
Tools to tailor a lookup table to a particular dataset.

Functions:
    compute_splines: Computes the likelihood for a set of configs from a
        precomputed lookup table.
"""
from __future__ import division

import numpy as np
from numba import njit

from pyrho.utility import get_table_idx, log_mult_coef


def _slow_sample(table, config, sample_size):
    """
    Gold standard for computing haplotype likelihoods.

    Explicitly marginalizes over each missing individual in turn.
    As a result, the runtime is exponential in the number of missing
    individuals. Should only be used for testing.
    Args:
        table: A matrix containing haplotype likelihoods.
        config: An integer array specifying the desired config.
        sample_size: The sample size for whcih table was computed.
    Returns:
        The log-likelihood of observing config evaluated at each value
        of rho in table.
    """
    hconf = np.array([[config[0], config[1], config[2]],
                      [config[3], config[4], config[5]],
                      [config[6], config[7], 0]])
    to_return = np.full(table.shape[1], np.NINF)
    free = sample_size - config.sum()
    if free > 0:
        for missing_left in range(2):
            for missing_right in range(2):
                to_add = np.zeros_like(config)
                to_add[missing_left * 3 + missing_right] = 1
                to_return = np.logaddexp(
                    to_return, _slow_sample(table,
                                            config + to_add,
                                            sample_size)
                )
        return to_return
    if hconf[0, -1] > 0:
        to_add0 = np.zeros_like(config)
        to_add0[2] = -1
        to_add0[0] = 1
        to_add1 = np.zeros_like(config)
        to_add1[2] = -1
        to_add1[1] = 1
    elif hconf[1, -1] > 0:
        to_add0 = np.zeros_like(config)
        to_add0[5] = -1
        to_add0[3] = 1
        to_add1 = np.zeros_like(config)
        to_add1[5] = -1
        to_add1[4] = 1
    elif hconf[-1, 0] > 0:
        to_add0 = np.zeros_like(config)
        to_add0[6] = -1
        to_add0[0] = 1
        to_add1 = np.zeros_like(config)
        to_add1[6] = -1
        to_add1[3] = 1
    elif hconf[-1, 1] > 0:
        to_add0 = np.zeros_like(config)
        to_add0[7] = -1
        to_add0[1] = 1
        to_add1 = np.zeros_like(config)
        to_add1[7] = -1
        to_add1[4] = 1
    else:
        this_idx = get_table_idx(hconf[0, 0], hconf[0, 1],
                                 hconf[1, 0], hconf[1, 1],
                                 sample_size)
        return table[this_idx, :]
    return np.logaddexp(
            _slow_sample(table, config + to_add0, sample_size),
            _slow_sample(table, config + to_add1, sample_size)
    )


@njit('float64(int64[:, :], int64, int64, int64, int64)', cache=True)
def _get_hap_comb(hconf, to_add00, to_add01, to_add10, to_add11):
    runtime = ((1 + min(to_add10, to_add00, hconf[-1, 0],
                        to_add00 + to_add10 - hconf[-1, 0]))
               * (1 + min(to_add01, to_add11, hconf[-1, 1],
                          to_add01 + to_add11 - hconf[-1, 1])))
    transpose_runtime = (
        (1 + min(to_add01, to_add00, hconf[0, -1],
                 to_add00 + to_add01 - hconf[0, -1]))
        * (1 + min(to_add10, to_add11, hconf[1, -1],
                   to_add10 + to_add11 - hconf[1, -1]))
    )
    if runtime > transpose_runtime:
        hconf = hconf.transpose()
        to_add01, to_add10 = to_add10, to_add01
    to_return = np.NINF
    imin = max(0, hconf[-1, 0] - to_add00)
    imax = min(to_add10, hconf[-1, 0])
    for i in range(imin, imax + 1):
        i_coef = np.NINF
        i_comb = log_mult_coef(np.array([i, hconf[-1, 0] - i]))
        jmin = max(0, hconf[-1, 1] - to_add11,
                   i + hconf[-1, 1] + hconf[1, -1] - to_add11 - to_add10)
        jmax = min(to_add00 + to_add01 + i - hconf[-1, 0] - hconf[0, -1],
                   to_add01, hconf[-1, 1])
        for j in range(jmin, jmax + 1):
            this_coef = log_mult_coef(
                np.array([to_add00 - hconf[-1, 0] + i, to_add01 - j])
            )
            this_coef += log_mult_coef(
                np.array([to_add11 - hconf[-1, 1] + j, to_add10 - i])
            )
            this_coef += log_mult_coef(np.array([
                to_add00 + to_add01 + i - j - hconf[-1, 0] - hconf[0, -1],
                to_add11 + to_add10 + j - i - hconf[-1, 1] - hconf[1, -1]])
            )
            this_coef += log_mult_coef(np.array([j, hconf[-1, 1] - j]))
            i_coef = np.logaddexp(i_coef, this_coef)
        to_return = np.logaddexp(to_return, i_coef + i_comb)
    return to_return


@njit('float64[:](float64[:, :], int64[:], int64[:])', cache=True)
def _get_hap_likelihood_fast_missing(table, subtable_sizes, config):
    hconf = np.array([[config[0], config[1], 0],
                      [config[3], config[4], 0],
                      [0, 0, 0]])
    this_size = hconf.sum()
    this_idx = get_table_idx(hconf[0, 0],
                             hconf[0, 1],
                             hconf[1, 0],
                             hconf[1, 1],
                             this_size)
    if this_size <= 1:
        return np.zeros_like(table[0, :])
    offset = 0 if this_size == 2 else subtable_sizes[this_size-3]
    log_comb = _get_hap_comb(hconf, 0, 0, 0, 0)
    return table[offset + this_idx, :] + log_comb


@njit('float64[:](float64[:, :], int64[:], int64)', cache=True)
def _get_hap_likelihood(table, config, sample_size):
    hconf = np.array([[config[0], config[1], config[2]],
                      [config[3], config[4], config[5]],
                      [config[6], config[7], 0]])
    to_return = np.full(table.shape[1], np.NINF)
    full = hconf[0:2, 0:2].sum()
    free = sample_size - full - hconf[-1, :].sum() - hconf[:, -1].sum()
    for to_add00 in range(free + hconf[0, -1] + hconf[-1, 0] + 1):
        a01_min = max(hconf[0, -1] - to_add00, 0)
        a01_max = min(free + hconf[0, -1] + hconf[-1, 1],
                      sample_size - full - to_add00 - hconf[1, -1])
        for to_add01 in range(a01_min, a01_max + 1):
            a10_min = max(hconf[-1, 0] - to_add00,
                          hconf[-1, 0] + hconf[0, -1] - to_add00 - to_add01,
                          0)
            a10_max = min(free + hconf[-1, 0] + hconf[1, -1],
                          sample_size - full - to_add00 - hconf[-1, 1],
                          sample_size - full - to_add00 - to_add01)
            for to_add10 in range(a10_min, a10_max + 1):
                to_add11 = sample_size - full - to_add00 - to_add01 - to_add10
                log_comb = _get_hap_comb(hconf, to_add00, to_add01, to_add10,
                                         to_add11)
                this_idx = get_table_idx(hconf[0, 0] + to_add00,
                                         hconf[0, 1] + to_add01,
                                         hconf[1, 0] + to_add10,
                                         hconf[1, 1] + to_add11,
                                         sample_size)
                this_ll = table[this_idx, :] + log_comb
                to_return = np.logaddexp(to_return, this_ll)
    return to_return


@njit('float64[:, :](float64[:, :], int64[:, :], int64)', cache=True)
def _get_splines_hap(table, configs, sample_size):
    to_return = np.zeros((configs.shape[0], table.shape[1]))
    for i in range(configs.shape[0]):
        to_return[i, :] = _get_hap_likelihood(table, configs[i, :],
                                              sample_size)
    return to_return


@njit('float64[:](float64[:, :], int64[:], int64[:], int64, boolean)', cache=True)
def _get_dip_likelihood(table, subtable_sizes, config, sample_size,
                        fast_missing):
    if fast_missing:
        gconf = np.array([[config[0], config[1], config[2], 0],
                          [config[4], config[5], config[6], 0],
                          [config[8], config[9], config[10], 0],
                          [0, 0, 0, 0]])

    else:
        gconf = np.array([[config[0], config[1], config[2], config[3]],
                          [config[4], config[5], config[6], config[7]],
                          [config[8], config[9], config[10], config[11]],
                          [config[12], config[13], config[14], 0]])
    to_return = np.full(table.shape[1], np.NINF)
    for k in range(gconf[1, 1] + 1):
        k_inv = gconf[1, 1] - k
        hap_conf = np.array([2*gconf[0, 0] + gconf[0, 1] + gconf[1, 0] + k,
                             2*gconf[0, 2] + gconf[0, 1] + gconf[1, 2] + k_inv,
                             2*gconf[0, -1] + gconf[1, -1],
                             2*gconf[2, 0] + gconf[2, 1] + gconf[1, 0] + k_inv,
                             2*gconf[2, 2] + gconf[2, 1] + gconf[1, 2] + k,
                             2*gconf[2, -1] + gconf[1, -1],
                             2*gconf[-1, 0] + gconf[-1, 1],
                             2*gconf[-1, 2] + gconf[-1, 1]])
        comb = log_mult_coef(np.array([k, k_inv]))
        if fast_missing:
            this_ll = _get_hap_likelihood_fast_missing(table, subtable_sizes,
                                                       hap_conf)
        else:
            this_ll = _get_hap_likelihood(table, hap_conf, sample_size)
        this_ll += comb
        to_return = np.logaddexp(to_return, this_ll)
    swaps = gconf[1, :].sum() + gconf[:, 1].sum() - gconf[1, 1]
    return swaps*np.log(2) + to_return


@njit('float64[:, :](float64[:, :], int64[:], int64[:, :], int64, boolean)', cache=True)
def _get_splines(table, subtable_sizes, configs, sample_size, fast_missing):
    ploidy = int(np.round(np.sqrt(configs.shape[1] + 1))) - 2
    if ploidy != 1 and ploidy != 2:
        raise NotImplementedError('Ploidies other than 1 and 2 '
                                  'are not implemented.')
    to_return = np.zeros((configs.shape[0], table.shape[1]))
    for i in range(configs.shape[0]):
        if ploidy == 1:
            if fast_missing:
                to_return[i, :] = _get_hap_likelihood_fast_missing(
                    table, subtable_sizes, configs[i, :]
                )
            else:
                to_return[i, :] = _get_hap_likelihood(table, configs[i, :],
                                                      sample_size)
        if ploidy == 2:
            to_return[i, :] = _get_dip_likelihood(table, subtable_sizes,
                                                  configs[i, :], sample_size,
                                                  fast_missing)
    return to_return


def compute_splines(configs, lookup_table, subtable_sizes, max_size, fast_missing):
    """
    Computes the log-likelihoods for a set of configs

    Args:
        configs: A num_configs by config_dim int array containing two-locus
            configurations. Haplotype configuations are encoded as
            (n_00, n_01, n_0*, n_10, n_11, n_1*, n_*0, n_*1), where n_ij is the
            number of haplotypes that have allele i at the first locus and
            allele j at the second locus.  * indicates missingness. Genotypes
            are encoded as (n_00, n_01, n_02, n_0*, n_10, n_11, n_12, n_1*,
            n_20, n_21, n_22, n_2*, n_*0, n_*1, n_*2).
        lookup_table: The numpy array containing the values of the
            pandas.DataFrame lookup table as computed by make_table.  If
            fast_missing is true, then this should contain the value from
            lookup tables for sample sizes 2...n concatenated.
        subtable_sizes: A numpy array of integers containing the (cummulative)
           sizes for each of the subtables making up table (if fast_missing
           is true).  If fast_missing is false, this is ignored.
        max_size: The number of haplotypes present in the dataset.
        fast_missing: A boolean which determines whether to use approximations
            to speed up inference.  If fast_mssing is true, then table and
            subtable_sizes need to be adjusted as described above.

    Returns:
        rho_values: A num_configs by num_rhos numpy array containing the
            log-likelihood of each config at each recombination rate.

    Raises:
        ArithmeticError: Error in computing splines.
    """
    rho_values = _get_splines(lookup_table, subtable_sizes, configs, max_size,
                              fast_missing)
    if np.any(np.isnan(rho_values)):
        raise ArithmeticError('Error in computing splines.')
    return rho_values
