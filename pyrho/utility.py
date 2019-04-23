"""
Numerical tools used throughout pyrho.

Classes:
    InterruptablePool: A wrapper for multiprocessing.Pool that handles
        keyborad interrupts more gracefully.

Functions:
    get_table_idx: Returns the lookup table index of a configuration.
    log_mult_coef: Computes the log of a multinomial coefficient.
    downsample: Computes a lookup table for a smaller sample size.
"""
from __future__ import division
import logging
import signal
from multiprocessing.pool import Pool

import numpy as np
from pandas import DataFrame
from numba import njit


def _pool_initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class InterruptablePool(Pool):
    """
    A wrapper for multiprocessing.Pool that handles keyboard interrupts.
    """

    def __init__(self, *args, **kwargs):
        kwargs['initializer'] = _pool_initializer
        super(InterruptablePool, self).__init__(*args, **kwargs)

    def map(self, func, iterable):
        try:
            return super(InterruptablePool, self).map(func, iterable)
        except KeyboardInterrupt:
            self.terminate()
            self.join()
            raise KeyboardInterrupt


@njit('int64(int64, int64, int64, int64, int64)', cache=True)
def get_table_idx(n00, n01, n10, n11, sample_size):
    """
    Returns the lookup table index for a two-locus configuration.

    Args:
        n00: Number of 00 haplotypes.
        n01: Number of 01 haplotypes.
        n10: Number of 10 haplotypes.
        n11: Number of 11 haplotypes.
        sample_size: The number of haplotypes for which the lookup table was
            computed.

    Returns:
       The index of the lookup table corresponding to the configuration. i.e.,
       lookup_table.values[get_table_idx(n00, n01, n10, n11, sample_size), :]
       will be the log-likelihood of the configuration.

    Raises:
        ValueError: Cannot obtain a table index for a negative count.
        ValueError: Cannot obtain a table index for the wrong n.
        ValueError: Cannot obtain a table index for a non-segregating allele.
    """
    if n00 < 0 or n01 < 0 or n10 < 0 or n11 < 0:
        raise ValueError('Cannot obtain a table index for a negative count.')
    if sample_size != n00 + n01 + n10 + n11:
        raise ValueError('Cannot obtain a table index for the wrong n.')
    if (
            n00 + n01 == sample_size
            or n10 + n11 == sample_size
            or n01 + n11 == sample_size
            or n10 + n00 == sample_size
    ):
        raise ValueError('Cannot obtain a table index for a non-segregating '
                         'allele.')
    if n00 < n11:
        n00, n11 = n11, n00
    if n01 < n10:
        n01, n10 = n10, n01
    if n11 + n01 > sample_size//2:
        n00, n01, n10, n11 = n01, n00, n11, n10
    i, j, k = n01+n11, n10+n11, n11
    return (j-k) + ((j-1) * j)//2 + (j-1) + round(((i-1)**3)/6 + (i-1)**2 +
                                                  5*(i-1)/6)


@njit('float64(int64[:])', cache=True)
def log_mult_coef(vector):
    """
    Computes the log of the multinomial coefficient of the integers in vector.

    Args:
        vector: An integer array.

    Returns:
        The log of the multinomial coefficient, which is the log of the
        factorial of the sum of vector divided by the product of
        the factorials of the entries of vector.

    Raises:
        ValueError: Negative value encountered in multinomial coefficient.
    """
    if vector.min() < 0:
        raise ValueError('Negative value encountered in multinomial '
                         'coefficient.')
    vector = vector[vector > 0]
    if vector.shape[0] <= 1:
        return 0.0
    vector = np.sort(vector[vector > 0])
    num = vector.sum()
    to_return = np.log(np.arange(vector[-1], num) + 1).sum()
    factorials = np.cumsum(np.log(np.arange(vector[-2]) + 1))
    for k in vector[:-1]:
        to_return = to_return - factorials[k-1]
    return to_return


def downsample(table, desired_size):
    """
    Computes a lookup table for a smaller sample size.

    Takes table and marginalizes over the last individuals to compute
    a lookup table for a sample size one smaller and repeats until reaching
    the desired_size.

    Args:
        table: A pandas.DataFrame containing a lookup table as computed by
            make_table.
        desired_size: The desired smaller sample size.

    Returns:
        A pandas.DataFrame containing a lookup table with sample size
        desired_size. The DataFrame is essentially the same as if make_table
        had been called with a smaller sample size.
    """
    first_config = table.index.values[0].split()
    curr_size = sum(map(int, first_config))
    rhos = table.columns
    curr_table = table.values
    while curr_size > desired_size:
        logging.info('Downsampling...  Currently at n = %d', curr_size)
        curr_table = _single_vec_downsample(curr_table, curr_size)
        curr_size = curr_size - 1
    halfn = curr_size // 2
    index = []
    idx = 0
    for i in range(1, halfn + 1):
        for j in range(1, i + 1):
            for k in range(j, -1, -1):
                n11 = k
                n10 = j - k
                n01 = i - k
                n00 = curr_size - i - j + k
                index += ['{} {} {} {}'.format(n00, n01, n10, n11)]
                idx += 1
    table = DataFrame(curr_table, index=index, columns=rhos)
    return table


@njit('float64[:, :](float64[:, :], int64)', cache=True)
def _single_vec_downsample(old_vec, sample_size):
    halfn = (sample_size - 1) // 2
    new_conf_num = (1 + halfn + halfn*(halfn - 1)*(halfn + 4)//6
                    + (halfn - 1)*(halfn + 2)//2)
    to_return = np.zeros((new_conf_num, old_vec.shape[1]))
    idx = 0
    for i in range(1, halfn+1):
        for j in range(1, i+1):
            for k in range(j, -1, -1):
                n11 = k
                n10 = j - k
                n01 = i - k
                n00 = sample_size - 1 - i - j + k
                add00 = get_table_idx(n00+1, n01, n10, n11, sample_size)
                add01 = get_table_idx(n00, n01+1, n10, n11, sample_size)
                add10 = get_table_idx(n00, n01, n10+1, n11, sample_size)
                add11 = get_table_idx(n00, n01, n10, n11+1, sample_size)
                to_return[idx, :] = np.logaddexp(old_vec[add00, :],
                                                 old_vec[add01, :])
                to_return[idx, :] = np.logaddexp(to_return[idx, :],
                                                 old_vec[add10, :])
                to_return[idx, :] = np.logaddexp(to_return[idx, :],
                                                 old_vec[add11, :])
                idx += 1
    return to_return
