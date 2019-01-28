from __future__ import division
from os.path import abspath, dirname, join as joinpath

import pytest
import numpy as np
from pandas import read_hdf
from scipy.special import binom
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from nose.tools import assert_raises

from pyrho.utility import get_table_idx, log_mult_coef, downsample
from pyrho.rho_splines import (_get_hap_likelihood,
                               _get_dip_likelihood,
                               _slow_sample,
                               _get_hap_comb)
from pyrho.optimizer import _rose_alg, _stitch
from pyrho.objective_function import _vec_map_splines, _vec_map_splines_d
from pyrho.hyperparameter_optimizer import _window_average
from pyrho.haplotype_reader import (_get_adjacency_matrix,
                                    _get_configs,
                                    _remove_non_segregating,
                                    parse_seqs_to_genos,
                                    parse_vcf_to_genos)
from pyrho.compute_r2 import _log_comb_factor, _compute_statistics


pytestmark = pytest.mark.filterwarnings("ignore")
THIS_DIR = dirname(abspath(__file__))


def test_log_mult_coef():
    assert np.allclose(np.log(binom(10, 2)), log_mult_coef(np.array([2, 8])))
    assert np.allclose(log_mult_coef(np.array([1, 1, 1])), np.log(6))
    assert np.allclose(log_mult_coef(np.array([0, 0, 0, 0])), 0)
    assert_raises(ValueError, log_mult_coef, np.array([-1]))


def test_get_table_idx():
    sample_size = 10
    halfn = sample_size // 2
    idx = 0
    for i in range(1, halfn + 1):
        for j in range(1, i + 1):
            for k in range(j, -1, -1):
                n11 = k
                n10 = j - k
                n01 = i - k
                n00 = sample_size - i - j + k
                assert idx == get_table_idx(n00, n01, n10, n11, sample_size)
                idx += 1
    assert_raises(ValueError, get_table_idx, 1, 1, 1, 1, 10)
    assert_raises(ValueError, get_table_idx, -1, 10, 1, 0, 10)
    assert_raises(ValueError, get_table_idx, 10, 0, 0, 0, 10)


def test_downsample():
    start_table = downsample(
        read_hdf(joinpath(THIS_DIR, 'n_15_test_table.hdf'), 'ldtable'), 10
    )
    target_table = read_hdf(joinpath(THIS_DIR, 'n_10_test_table.hdf'),
                            'ldtable')
    assert np.allclose(start_table.values, target_table.values)
    assert np.all(start_table.columns == target_table.columns)
    assert np.all(start_table.index == target_table.index)


def test_get_hap_comb():
    hconf = np.zeros((3, 3), dtype=int)
    to_add00, to_add01, to_add10, to_add11 = 6, 6, 7, 5
    hconf[-1, 0] = 10
    hconf[-1, 1] = 10
    hconf[0, -1] = 2
    hconf[1, -1] = 2
    hconf_copy = np.copy(hconf)
    _get_hap_comb(hconf, to_add00, to_add01, to_add10, to_add11)
    assert to_add00 == 6
    assert to_add01 == 6
    assert to_add10 == 7
    assert to_add11 == 5
    assert np.all(hconf == hconf_copy)


def test_hap_likelihood():
    big_table = read_hdf(joinpath(THIS_DIR, 'n_15_test_table.hdf'),
                         'ldtable')
    small_table = read_hdf(joinpath(THIS_DIR, 'n_10_test_table.hdf'),
                           'ldtable')
    big_like = _get_hap_likelihood(big_table.values,
                                   np.array([5, 3, 0, 2, 0, 0, 0, 0]),
                                   15)
    small_like = _get_hap_likelihood(small_table.values,
                                     np.array([5, 3, 0, 2, 0, 0, 0, 0]), 10)
    assert np.allclose(small_like,
                       small_table.values[get_table_idx(5, 3, 2, 0, 10), :])
    assert np.allclose(big_like, small_like)

    full_like = np.logaddexp(
        _get_hap_likelihood(small_table.values,
                            np.array([5, 3, 0, 2, 0, 0, 0, 0]),
                            10),
        _get_hap_likelihood(small_table.values,
                            np.array([6, 2, 0, 2, 0, 0, 0, 0]),
                            10)
    )
    partial_like = _get_hap_likelihood(small_table.values,
                                       np.array([5, 2, 1, 2, 0, 0, 0, 0]),
                                       10)
    assert np.allclose(full_like, partial_like)

    partial_like = _get_hap_likelihood(small_table.values,
                                       np.array([1, 2, 1, 2, 0, 0, 1, 2]),
                                       10)
    partial_like_big = _get_hap_likelihood(big_table.values,
                                           np.array([1, 2, 1, 2, 0, 0, 1, 2]),
                                           15)
    truth = _slow_sample(small_table.values,
                         np.array([1, 2, 1, 2, 0, 0, 1, 2]), 10)

    assert np.allclose(partial_like, partial_like_big)
    assert np.allclose(partial_like, truth)

    full_like = np.logaddexp(
        _get_hap_likelihood(small_table.values,
                            np.array([6, 2, 0, 2, 0, 0, 0, 0]),
                            10),
        _get_hap_likelihood(small_table.values,
                            np.array([5, 2, 0, 3, 0, 0, 0, 0]),
                            10)
    )

    partial_like = _get_hap_likelihood(small_table.values,
                                       np.array([5, 2, 0, 2, 0, 0, 1, 0]),
                                       10)
    truth = _slow_sample(small_table.values,
                         np.array([5, 2, 0, 2, 0, 0, 1, 0]), 10)
    assert np.allclose(full_like, partial_like)
    assert np.allclose(partial_like, truth)
    partial_like = _get_hap_likelihood(small_table.values,
                                       np.array([0, 0, 3, 0, 0, 3, 1, 1]),
                                       10)
    truth = _slow_sample(small_table.values,
                         np.array([0, 0, 3, 0, 0, 3, 1, 1]), 10)
    assert np.allclose(partial_like, truth)


def test_dip_likelihood():
    big_table = read_hdf(joinpath(THIS_DIR, 'n_15_test_table.hdf'),
                         'ldtable').values
    small_table = read_hdf(joinpath(THIS_DIR, 'n_10_test_table.hdf'),
                           'ldtable').values
    big_like = _get_dip_likelihood(
        big_table,
        np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        15
    )
    small_like = _get_dip_likelihood(
        small_table,
        np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        10
    )
    assert np.allclose(small_like, big_like)
    big_like = _get_dip_likelihood(
        big_table,
        np.array([1, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        15
    )
    small_like = _get_dip_likelihood(
        small_table,
        np.array([1, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        10
    )
    assert np.allclose(small_like, big_like)
    big_like = _get_dip_likelihood(
        big_table,
        np.array([1, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0]),
        15
    )
    small_like = _get_dip_likelihood(
        small_table,
        np.array([1, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0]),
        10
    )
    assert np.allclose(small_like, big_like)


def test_rose_algorithm():
    delta_matrix = dok_matrix((10, 11))
    delta_matrix[[np.arange(10)], [np.arange(1, 11)]] = 1
    delta_matrix[[np.arange(10)], [np.arange(10)]] = -1
    delta_matrix = delta_matrix.tocsr()
    rrt = delta_matrix.dot(delta_matrix.transpose())
    u = np.random.normal(size=10)
    assert np.allclose(spsolve(rrt, u), _rose_alg(u))


def test_stitch():
    to_stitch = [np.arange(5), np.arange(3, 8), np.arange(6, 11)]
    assert np.all(_stitch(to_stitch, 1) == np.arange(11))
    to_stitch = [np.arange(5), np.arange(5, 10)]
    assert np.all(_stitch(to_stitch, 0) == np.arange(10))


def test_map_splines():
    rhos = np.array([0.25, 0.5, 0.0, 2.0])
    rho_grid = np.array([0., 1.])
    values = np.array([[10., 0.],
                       [10., 0.],
                       [10., 0.],
                       [10., 0.]])
    assert np.allclose(_vec_map_splines(rhos, values, rho_grid),
                       np.array([7.5, 5.0, 10.0, 0.0]))
    assert np.allclose(_vec_map_splines_d(rhos, values, rho_grid),
                       np.array([-10.0, -10.0, -10.0, 0.0]))


def test_window_average():
    x = np.random.normal(size=10)
    x_mean = np.mean(x)
    assert np.allclose(x_mean, _window_average((x, 10)))


def test_get_adjacency_matrix():
    a_mat = _get_adjacency_matrix(2, 10)
    a_mat_true = np.zeros((3, 9))
    a_mat_true[0, :] = 1
    a_mat_true[1, :] = np.arange(9)
    a_mat_true[2, :] = np.arange(9)
    assert np.all(a_mat == a_mat_true)

    a_mat = _get_adjacency_matrix(3, 10)
    a_mat_true = np.zeros((3, 25))
    a_mat_true[0, :] = 1
    a_mat_true[1, :] = np.array([0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9,
                                 10, 11, 11, 12, 13, 13, 14, 15, 15, 16])
    a_mat_true[2, :] = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,
                                 5, 5, 6, 6, 6, 7, 7, 7, 8, 8])
    assert np.all(a_mat == a_mat_true)


def test_get_configs():
    gts = np.array([[0, 0, 0, 1, 1, 1, -1, -1, -1],
                    [0, 1, -1, 0, 1, -1, 0, 1, -1]])
    confs = _get_configs(gts, 2, 1)
    assert np.all(confs == np.ones(8))

    dip_confs = _get_configs(gts, 2, 2)
    assert np.all(dip_confs == np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
                                         1, 0]))

    gts[1, 0] = 2
    dip_confs = _get_configs(gts, 2, 2)
    assert np.all(dip_confs == np.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
                                         1, 0]))


def test_remove_non_segregating():
    gts = np.ones((10, 100))
    pos = np.arange(100)
    gts[0, 0] = 0
    seg_gts, seg_pos = _remove_non_segregating(gts, pos, 1)
    assert np.all(seg_gts.flatten() == gts[:, 0])
    assert np.all(seg_pos == 0)

    gts = 1 - gts
    seg_gts, seg_pos = _remove_non_segregating(gts, pos, 1)
    assert np.all(seg_gts.flatten() == gts[:, 0])
    assert np.all(seg_pos == 0)

    gts[1, :] = -1
    seg_gts, seg_pos = _remove_non_segregating(gts, pos, 1)
    assert np.all(seg_gts.flatten() == gts[:, 0])
    assert np.all(seg_pos == 0)

    gts = np.ones((10, 100)) * 2
    gts[0, 0] = 0
    seg_gts, seg_pos = _remove_non_segregating(gts, pos, 2)
    assert np.all(seg_gts.flatten() == gts[:, 0])
    assert np.all(seg_pos == 0)

    gts[0, 1] = -1
    seg_gts, seg_pos = _remove_non_segregating(gts, pos, 2)
    assert np.all(seg_gts.flatten() == gts[:, 0])
    assert np.all(seg_pos == 0)


def test_parse_seqs_to_genos():
    gts, locs = parse_seqs_to_genos(joinpath(THIS_DIR, 'test_dna.fa'), 1, '')
    expected = np.array([[1, 1, 0,  0],
                         [1, 0, 0, 1]])
    expected_locs = np.array([0, 1])
    assert np.all(gts == expected)
    assert np.all(locs == expected_locs)

    assert_raises(IOError, parse_seqs_to_genos,
                  joinpath(THIS_DIR, 'test_dna.fa'), 2, '')

    gts, locs = parse_seqs_to_genos(joinpath(THIS_DIR, 'test.fa'), 1, '')
    expected = np.array([[0, 0, 1, 1, -1],
                         [0, 1, 0, 1, -1]])
    assert np.all(gts == expected)
    assert np.all(locs == expected_locs)

    assert_raises(IOError,
                  parse_seqs_to_genos,
                  joinpath(THIS_DIR, 'sites.txt'),
                  1,
                  joinpath(THIS_DIR, 'locs.txt'))
    gts, locs = parse_seqs_to_genos(joinpath(THIS_DIR, 'sites.txt'), 2,
                                    joinpath(THIS_DIR, 'locs.txt'))
    expected = np.array([[0, 1, 0, 2],
                         [0, 0, 1, 2]])
    expected_locs = np.array([25.0, 77.0])
    assert np.all(gts == expected)
    assert np.allclose(locs, expected_locs)


def test_parse_vcf_to_genos():
    positions = np.array([199, 320])
    gts = np.array([[0, 0, 0, 1, 1, 1, -1, -1],
                    [0, 1, -1, 0, 1, -1, 0, 1]])
    genos, locs = parse_vcf_to_genos(joinpath(THIS_DIR, 'short.vcf.gz'),
                                     ploidy=1)
    assert np.all(positions == locs), genos
    assert np.all(gts == genos)

    gts = np.array([[0, 1, 2, -1],
                    [1, -1, -1, 1]])
    genos, locs = parse_vcf_to_genos(joinpath(THIS_DIR, 'short.vcf.gz'),
                                     ploidy=2)
    assert np.all(positions == locs), genos[0, :]
    assert np.all(gts == genos)


def test_log_comb_factor():
    def log_num_swaps(n00, n01, n10, n11):
        confs = [(n00, n01, n10, n11),
                 (n00, n10, n01, n11),
                 (n11, n10, n01, n00),
                 (n11, n01, n10, n00),
                 (n01, n00, n11, n10),
                 (n01, n11, n00, n10),
                 (n10, n00, n11, n01),
                 (n10, n11, n00, n01)]
        return np.log(len(set(confs)))

    lcf = _log_comb_factor(np.array([[2, 2],
                                     [2, 2]]))
    lcf_true = log_mult_coef(np.array([2, 2, 2, 2]))
    assert np.allclose(lcf, lcf_true)

    lcf = _log_comb_factor(np.array([[3, 2],
                                     [2, 2]]))

    lcf_true = (log_mult_coef(np.array([3, 2, 2, 2]))
                + log_num_swaps(3, 2, 2, 2))
    assert np.allclose(lcf, lcf_true)

    lcf = _log_comb_factor(np.array([[3, 2],
                                     [2, 3]]))
    lcf_true = (log_mult_coef(np.array([3, 2, 2, 3]))
                + log_num_swaps(3, 2, 2, 3))
    assert np.allclose(lcf, lcf_true), np.exp(log_num_swaps(3, 2, 2, 3))

    lcf = _log_comb_factor(np.array([[1, 2],
                                     [3, 4]]))
    lcf_true = (log_mult_coef(np.array([1, 2, 3, 4]))
                + log_num_swaps(1, 2, 3, 4))
    assert np.allclose(lcf, lcf_true)

    lcf = _log_comb_factor(np.array([[2, 2],
                                     [3, 3]]))
    lcf_true = (log_mult_coef(np.array([2, 2, 3, 3]))
                + log_num_swaps(2, 2, 3, 3))
    assert np.allclose(lcf, lcf_true)


def test_compute_statistics():
    table = downsample(read_hdf(joinpath(THIS_DIR, 'n_10_test_table.hdf'),
                                'ldtable'), 2).values
    means = _compute_statistics(np.array([], dtype=np.float64),
                                True,
                                0.0,
                                2,
                                table)
    assert np.allclose(means, 1.0)
