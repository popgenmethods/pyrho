"""
The pyrho fused-LASSO objective function.

Classes:
    RhomapObjective: The objective function for a given dataset.
"""
from __future__ import division

import numpy as np
from scipy.sparse import dok_matrix


class RhomapObjective(object):
    """
    The fused-LASSO objective function for a given dataset.

    Attributes:
        block_penalty: The strength of the L1 regularization penalty.
        adjacency_map: A binary scipy.sparse.csr_matrix indicating which
            recombination rates are between each two-locus configurations.
        lengths: The physical distance between pairs of adjacent SNPs.
        rho_grid: An array of recombination rates.
        rho_values: A table containing the two-locus likelihoods for the
            two-locus configurations in adjacency_map that for the
            recombination rates in rho_grid.
        delta_matrix: A scipy.sparse.csr_matrix that maps a vector of
            recombination rates to a vector one dimension lower that
            contains the difference in adjacent recombination rates.
        rrt: The scipy.sparse.csr_matrix delta_matrix dot delta_matrix.T
    """
    def __init__(
            self, rho_values, rho_grid, adjacency_map, lengths, block_penalty
    ):
        """Inits RhomapObjective and constructs delta_matrix and rrt"""
        self.block_penalty = block_penalty
        self.adjacency_map = adjacency_map
        self.lengths = lengths
        self.rho_values = rho_values
        self.rho_grid = rho_grid
        num_rhos = adjacency_map.shape[1]
        delta_matrix = dok_matrix((num_rhos - 1, num_rhos))
        delta_matrix[[np.arange(num_rhos - 1)],
                     [np.arange(1, num_rhos)]] = 1
        delta_matrix[[np.arange(num_rhos - 1)],
                     [np.arange(num_rhos - 1)]] = -1
        self.delta_matrix = delta_matrix.tocsr()
        self.rrt = self.delta_matrix.dot(self.delta_matrix.transpose())

    def negative_loglihood_one_rho(self, rho):
        """Returns the neg. log-like. of a reco. map with constant rate rho"""
        return self.negative_loglihood(
            np.full(self.adjacency_map.shape[1], rho)
        )

    def negative_loglihood(self, rhos):
        """Returns the neg. log-like. of a reco. map rhos"""
        rhos = np.exp(rhos)
        rhos_between = self.adjacency_map.dot(self.lengths*rhos)
        log_likelihood = _vec_map_splines(rhos_between, self.rho_values,
                                          self.rho_grid).sum()
        return -log_likelihood

    def value(self, rhos):
        """Returns the value of the objective function for a reco. map rhos"""
        log_likelihood = self.negative_loglihood(rhos)
        l1_penalty = (self.block_penalty
                      * np.abs(self.delta_matrix.dot(rhos))).sum()
        return log_likelihood + l1_penalty

    def value_and_grad(self, rhos):
        """Returns value of the obj. function and the neg. log-like.  grad."""
        log_likelihood, grad = self.negative_loglihood_and_grad(rhos)
        l1_penalty = (self.block_penalty
                      * np.abs(self.delta_matrix.dot(rhos))).sum()
        return log_likelihood + l1_penalty, grad

    def negative_loglihood_and_grad(self, rhos):
        """Returns the neg. log-like. and its gradient"""
        rhos = np.exp(rhos) * self.lengths
        rhos_between = self.adjacency_map.dot(rhos)
        likes = -_vec_map_splines(rhos_between, self.rho_values,
                                  self.rho_grid).sum()
        like_grad = -_vec_map_splines_d(rhos_between, self.rho_values,
                                        self.rho_grid)
        grad = rhos * (self.adjacency_map.transpose().dot(like_grad))
        return likes, grad


def _vec_map_splines(rhos, values, rho_grid):
    rho_idx = np.searchsorted(rho_grid, rhos, side='right') - 1
    num_rhos = rho_grid.shape[0]
    valid = rho_idx < (num_rhos - 1)
    rho_idx_valid = rho_idx[valid]
    rho_valid = rhos[valid]
    to_return = np.copy(values[:, -1])
    left_vals = values[valid, rho_idx_valid]
    right_vals = values[valid, rho_idx_valid + 1]
    left_dist = rho_valid - rho_grid[rho_idx_valid]
    right_dist = rho_grid[rho_idx_valid + 1] - rho_valid
    to_return[valid] = ((left_vals * right_dist + right_vals * left_dist)
                        / (left_dist + right_dist))
    return to_return


def _vec_map_splines_d(rhos, values, rho_grid):
    rho_idx = np.searchsorted(rho_grid, rhos, side='right') - 1
    num_rhos = rho_grid.shape[0]
    valid = rho_idx < (num_rhos - 1)
    rho_valid = rho_idx[valid]
    to_return = np.zeros_like(rhos)
    to_return[valid] = ((values[valid, rho_valid + 1]
                         - values[valid, rho_valid])
                        / (rho_grid[rho_valid + 1]
                            - rho_grid[rho_valid]))
    return to_return
