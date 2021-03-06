"""
Tools for parsing MSMC and smc++ output

Functions:
    read_msmc: Parses MSMC format files.
    read_smcpp: Parses smc++ format files.
    decimate_sizes: Simplifies size histories.
"""
import numpy as np
import pandas as pd


def read_msmc(fname, mut_rate):
    """
    Parses an MSMC format file into sizes and times.

    Arguments:
        fname: Path to msmc format file.
        mut_rate: Per-generation mutation rate

    Returns:
        A tuple (sizes, times) containing lists of the population sizes and
        epoch break points.
    """
    size_history = pd.read_csv(fname,
                               delim_whitespace=True,
                               dtype=np.float64,
                               header=0)
    col_names = ['time_index', 'left_time_boundary',
                 'right_time_boundary', 'lambda']
    for col, expected in zip(size_history.columns, col_names):
        if not col.startswith(expected):
            raise IOError('MSMC file does not containa a proper header!\n'
                          'time_index\tleft_time_boundary\t'
                          'right_time_boundary\tlambda')

    if len(size_history.columns) != 4:
        raise IOError('MSMC file either contains more than one '
                      'population or is not formatted correctly.')
    size_history.rename(columns=dict(zip(size_history.columns, col_names)),
                        inplace=True)
    if not np.allclose(size_history['left_time_boundary'][1:],
                       size_history['right_time_boundary'][:-1]):
        raise IOError('Ends of epochs in MSMC do not line up with '
                      'starts of subsequent epochs.')
    times = np.array(size_history['left_time_boundary'])[1:] / mut_rate
    if times[0] <= 0.:
        raise IOError('Times must be strictly positive')
    sizes = (1. / np.array(size_history['lambda'])) / (2. * mut_rate)
    if np.any(sizes <= 0):
        raise IOError('Sizes must be strictly positive')
    return sizes.tolist(), times.tolist()


def read_smcpp(fname):
    """
    Parses an smc++ plot csv file into sizes and times.

    Arguments:
        fname: Path to smc++ csv format file.

    Returns:
        A tuple (sizes, times) containing lists of the population sizes and
        epoch break points.
    """

    size_history = pd.read_csv(fname)
    times = size_history['x'][1:]
    sizes = size_history['y']
    return sizes.tolist(), times.tolist()


def decimate_sizes(sizes, times, rel_tol, anc_size):
    """
    Simplifies a population size history by combining close epochs.

    Arguments:
        sizes: A list of population sizes for each epoch.
        times: A list of epoch break points.
        rel_tol: Guarantee that all sizes in the new size history are within
                 rel_tol of the sizes in the old size history.
        anc_size: Force the most ancient population size to be anc_size, for
                  reusing ldpop stationary distributions.

    Returns:
        A tuple (sizes, times) containing lists of the population sizes and
        epoch break points for the new decimated size history.
    """
    rel_tol = max([rel_tol, 1e-6])
    sizes = np.array(sizes)
    if not anc_size:
        anc_size = sizes[-1]
    times = np.concatenate([np.zeros(1), np.array(times)])
    lens = np.diff(times)
    weighted_lens = lens / sizes[:-1]
    down_sizes = []
    down_times = []
    prev_idx = 0
    curr_idx = 1
    prev_size = None
    while curr_idx <= lens.shape[0]:
        harmonic_size = lens[prev_idx:curr_idx].sum()
        harmonic_size /= weighted_lens[prev_idx:curr_idx].sum()
        rel_diffs = harmonic_size - sizes[prev_idx:curr_idx]
        rel_diffs /= sizes[prev_idx:curr_idx]
        if np.any(np.abs(rel_diffs) > rel_tol):
            down_sizes.append(prev_size)
            down_times.append(times[curr_idx-1])
            prev_idx = curr_idx - 1
            prev_size = sizes[prev_idx]
        else:
            prev_size = harmonic_size
        curr_idx += 1
    if np.abs((prev_size - anc_size) / prev_size) > rel_tol:
        down_times.append(times[curr_idx - 1])
        down_sizes.append(prev_size)
    return down_sizes + [anc_size], down_times
