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
from pyrho.size_reader import read_msmc, read_smcpp, decimate_sizes

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


def test_read_msmc():
    true_sizes = [0.00086526447673996, 9.79088625144171e-05,
                  0.000171382542974173, 0.000457984501804459,
                  0.000272927253969727, 0.000131071906047658,
                  6.82882858274492e-05, 4.10030998343475e-05,
                  2.79371748811273e-05, 2.12348489352847e-05,
                  1.99020025394955e-05, 1.99020025394955e-05,
                  2.2919799039202e-05, 2.2919799039202e-05,
                  2.98841094236551e-05, 2.98841094236551e-05,
                  4.04164511122607e-05, 4.04164511122607e-05,
                  5.57810348943842e-05, 5.57810348943842e-05,
                  8.00343187158654e-05, 8.00343187158654e-05,
                  0.000116691016192045, 0.000116691016192045,
                  0.000167278907468, 0.000167278907468,
                  0.000226324223028942, 0.000226324223028942,
                  0.000287970327537451, 0.000287970327537451,
                  0.000345006037605658, 0.000345006037605658,
                  0.000392215310516861, 0.000392215310516861,
                  0.000424300540558889, 0.000424300540558889,
                  0.000447447313078885, 0.000447447313078885,
                  0.000841088166225936, 0.000841088166225936]
    true_times = [1.58858e-06, 3.21843e-06, 4.89174e-06,
                  6.61091e-06, 8.3785e-06, 1.01973e-05, 1.20705e-05,
                  1.40013e-05, 1.59934e-05, 1.80508e-05, 2.01779e-05,
                  2.23798e-05, 2.46617e-05, 2.70297e-05, 2.94906e-05,
                  3.2052e-05, 3.47225e-05, 3.75116e-05, 4.04305e-05,
                  4.34919e-05, 4.67103e-05, 5.01028e-05, 5.36893e-05,
                  5.74932e-05, 6.15427e-05, 6.58717e-05, 7.05216e-05,
                  7.5544e-05, 8.10035e-05, 8.69838e-05, 9.35947e-05,
                  0.000100985, 0.000109364, 0.000119036, 0.000130476,
                  0.000144477, 0.000162528, 0.000187969, 0.000231461]
    sizes, times = read_msmc(joinpath(THIS_DIR, 'msmc_test.final.txt'),
                             1.0)
    assert np.allclose(sizes, true_sizes)
    assert np.allclose(times, true_times)

    sizes, times = read_msmc(joinpath(THIS_DIR, 'msmc_test.final.txt'),
                             1.25e-8)
    assert np.allclose(sizes, np.array(true_sizes) / 1.25e-8)
    assert np.allclose(times, np.array(true_times) / 1.25e-8)


def test_read_smcpp():
    true_sizes_start = [138482.84333082315,
                        138482.84333082315,
                        139331.82583178935]
    true_sizes_end = [19408.187247411068,
                      20959.43140840318,
                      23058.569473392425]
    true_times_start = [50.0, 53.97505585700569,
                        58.2661330953377]
    true_times_end = [83485.36048509754,
                      90122.53990850793,
                      97287.38251073883]
    sizes, times = read_smcpp(joinpath(THIS_DIR, 'ACB_pop_sizes.csv'))
    assert np.allclose(sizes[0:3], true_sizes_start)
    assert np.allclose(sizes[-3:], true_sizes_end)
    assert np.allclose(times[0:3], true_times_start)
    assert np.allclose(times[-3:], true_times_end)


def test_decimate_sizes():
    sizes, times = read_smcpp(joinpath(THIS_DIR, 'ACB_pop_sizes.csv'))
    new_sizes, new_times = decimate_sizes(sizes, times, 0.0, None)
    assert np.allclose(sizes[1:], new_sizes)
    assert np.allclose(times[1:], new_times)
    new_sizes, new_times = decimate_sizes(sizes, times, 0.0, 1.0)
    assert np.allclose(sizes[1:-1], new_sizes[:-1])
    assert np.allclose(1.0, new_sizes[-1])
    assert np.allclose(times[1:], new_times)
    new_sizes, new_times = decimate_sizes(sizes, times, 0.25, None)
    new_idx = 0
    for idx, t in enumerate(times):
        if t > new_times[new_idx]:
            new_idx += 1
            assert new_times[new_idx] >= t
        rel_error = np.abs((sizes[idx] - new_sizes[new_idx]))
        rel_error /= sizes[idx]
        assert rel_error < 0.25
