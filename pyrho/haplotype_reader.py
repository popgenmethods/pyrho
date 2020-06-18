"""
A collection of methods for reading and processing data.

Functions:
    parse_vcf_to_genos: Uses cyvcf2 to parse VCF format data into genotypes.
    parse_seqs_to_genos: Parses fasta format or LDHat sites and locs data.
    genos_to_configs: Converts genotypes into an array of two-locus configs.
"""
from __future__ import division
import logging
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
from cyvcf2 import VCF


@njit('int64(int64, int64)', cache=True)
def _get_adj_size(window_size, haplen):
    if window_size < haplen:
        return ((window_size - 1) * (haplen-window_size)
                + ((window_size - 1) * window_size) // 2)
    else:
        return (haplen * (haplen-1)) // 2


@njit('int64[:, :](int64, int64)', cache=True)
def _get_adjacency_matrix(window_size, haplen):
    conf_indices = []
    rho_indices = []
    conf_idx = 0
    total = 0
    for idx1 in range(haplen-1):
        for idx2 in range(idx1 + 1, min((idx1+window_size, haplen))):
            for idx3 in range(idx1, idx2):
                conf_indices.append(conf_idx)
                rho_indices.append(idx3)
                total += 1
            conf_idx += 1
    to_return = np.zeros((3, total), dtype=np.int64)
    to_return[0, :] = 1
    to_return[1, :] = conf_indices
    to_return[2, :] = rho_indices
    return to_return


@njit('int64[:, :](int64[:, :], int64, int64)', cache=True)
def _get_configs(genotypes, window_size, ploidy):
    haplen = genotypes.shape[0]
    sample_size = genotypes.shape[1]
    adj_size = _get_adj_size(window_size, haplen)
    to_return = np.zeros((adj_size, (ploidy + 2)**2 - 1), dtype=np.int64)
    conf_idx = 0
    for idx1 in range(haplen):
        for idx2 in range(idx1+1, min((idx1+window_size, haplen))):
            for gen_idx in range(sample_size):
                left_gt = genotypes[idx1, gen_idx]
                if left_gt == -1:
                    left_gt = ploidy + 1
                right_gt = genotypes[idx2, gen_idx]
                if right_gt == -1:
                    right_gt = ploidy + 1
                gtype = (ploidy + 2) * left_gt + right_gt
                if gtype < to_return.shape[1]:
                    to_return[conf_idx, gtype] += 1
            conf_idx += 1
    return to_return


def genos_to_configs(genotypes, window_size, ploidy):
    """
    Converts a genotype array into an array of two-locus configs.

    Args:
        genotypes: A num_loci by num_samples array containing the data.
        window_size: Consider all pairs within windows of this size.
        ploidy: The ploidy of genotypes (i.e. 1 for phased and 2 for
            unphased data).

    Returns:
        A tuple (configs, adjacency_matix) as follows.
        configs: All two-locus configurations formed by considering pairs of
            loci within windows of size window_size
        adjacency_matrix: A binary num_configs by num_recombination_rate
            array indicating which recombination rates are between the loci
            loci that make up each config.
    """
    genotypes = np.array(genotypes, dtype=np.int64)
    haplen = genotypes.shape[0]

    adj_mat_coord = _get_adjacency_matrix(window_size, haplen)
    adj_size = _get_adj_size(window_size, haplen)
    adjacency_matrix = csr_matrix(
        (adj_mat_coord[0, :], (adj_mat_coord[1, :], adj_mat_coord[2, :])),
        shape=(adj_size, haplen - 1), dtype=int
    )
    configs = _get_configs(genotypes, window_size, ploidy)

    return configs, adjacency_matrix


def _read_locs(locs_filename):
    with open(locs_filename) as locs_fh:
        header = locs_fh.readline().split()
        if header[2] != 'L':
            raise NotImplementedError(
                'Third entry of header of locs file must be "L". '
                'This designates inference of cross-over rate. '
                'Inference of gene conversion is not implemented.'
            )
        num_loci = int(header[0])
        locus_length = float(header[1])
        locs = list(map(float, locs_fh.readline().split()))
        if len(locs) != num_loci:
            raise IOError('Different number of loci in '
                          'header and body of locs file.')
        if locs[-1] > locus_length:
            raise IOError('Last locus in locs file is beyond the end '
                          'of the sequence.')
    return np.array(locs)


def _read_fasta(fasta_filename, ploidy):
    with open(fasta_filename) as fasta_fh:
        dna_map = {'A': ploidy + 1, 'a': ploidy + 1,
                   'C': ploidy + 2, 'c': ploidy + 2,
                   'G': ploidy + 3, 'g': ploidy + 3,
                   'T': ploidy + 4, 't': ploidy + 4}

        def _fasta_char_to_int(char):
            if ploidy > 1:
                try:
                    if int(char) > ploidy:
                        raise IOError('Genotypes must be encoded as digits. '
                                      'Missingness should be encoded as "N".')
                    return int(char)
                except ValueError:
                    if char != 'N':
                        raise IOError('Genotypes must be encoded as digits. '
                                      'Missingness should be encoded as "N".')
                    return -1
            if ploidy == 1:
                try:
                    if int(char) > ploidy:
                        raise IOError('Alleles must be encoded as digits or '
                                      'DNA bases.  Missingness should be '
                                      'encoded as "N".')
                    return int(char)
                except ValueError:
                    try:
                        return dna_map[char]
                    except KeyError:
                        return -1

        genos = []
        for line in fasta_fh:
            if line.startswith('>'):
                try:
                    genos[-1] = list(chain().from_iterable(genos[-1]))
                except IndexError:
                    pass
                genos.append([])
            else:
                try:
                    genos[-1].append(map(_fasta_char_to_int,
                                         line.strip()))
                except IndexError:
                    continue
        genos[-1] = list(chain().from_iterable(genos[-1]))
    return np.array(genos)


def _remove_non_segregating(genos, locs, ploidy):
    num_not_missing = (genos.shape[0] - (genos == -1).sum(axis=0)) * ploidy
    counts = (genos * (genos != -1)).sum(axis=0)
    keep = (counts > 0) * (counts < num_not_missing)
    return genos[:, keep], locs[keep]


def parse_seqs_to_genos(fasta_filename, ploidy, locs_filename=None):
    """
    Parses a fasta file or LDHat-style sites and locs file to a genotype array.

    Args:
        fasta_filename: A str containing the name of either a fasta file or an
            LDHat-style sites file. May contain either DNA (A, C, G, T) or
            genotypes (0/1 for phased data, 0/1/2 for unphased data). Missing
            Missing data should be encoded as "N". NOTE: LDHat uses the
            convention that 0 = Hom. Ref., 1 = Hom. Alt., and 2 = Het. We adopt
            the more sensible convention that 0 = Hom. Ref., 1 = Het.,
            and 2 = Hom. Alt.
        ploidy: The ploidy of the data, 1 for phased data and 2 for unphased
            data.
        locs_filename: A str containing the name of a LDHat-style locs file
            that contains the genomic location of the alleles in
            fasta_filename. If None locations are assumed to be
            0, 1, ..., num_loci - 1

    Returns:
        A tuple of (genos, locs) as follows.
        genos: A num_loci by num_samples array containing the data with -1
            encoding missingness. genos only contains segregating sites.
        locs: A num_loci length array containing the genomic locations of the
            loci in genos.

    Raises:
        IOError: Alleles cannot be encoded as both digits and DNA.
        IOError: SNP locations must be strictly increasing.
        NotImplementedError: Third entry of header of locs file must be "L".
    """
    genos = _read_fasta(fasta_filename, ploidy)
    if (
            np.any(genos[genos >= 0] > ploidy)
            and np.any(genos[genos >= 0] <= ploidy)
    ):
        raise IOError('Alleles cannot be encoded as both digits and DNA.')
    keep = np.ones(genos.shape[1], dtype=bool)
    if np.any(genos[genos >= 0] > ploidy):
        a_present = np.any(genos == ploidy + 1, axis=0).astype(int)
        c_present = np.any(genos == ploidy + 2, axis=0).astype(int)
        g_present = np.any(genos == ploidy + 3, axis=0).astype(int)
        t_present = np.any(genos == ploidy + 4, axis=0).astype(int)
        keep = (a_present + c_present + g_present + t_present) <= 2
        genos[genos >= 0] = (genos < genos.max(axis=0))[genos >= 0]
    if locs_filename:
        locs = _read_locs(locs_filename)
    else:
        locs = np.arange(genos.shape[1])
    genos = genos[:, keep]
    locs = locs[keep]
    genos, locs = _remove_non_segregating(genos, locs, ploidy)
    genos = genos.transpose()
    if np.any(np.diff(locs) <= 0.0):
        raise IOError('SNP locations must be strictly increasing.')
    return genos, locs


def parse_vcf_to_genos(vcf_filename, ploidy, pass_str=None):
    """
    Parses VCFs, bgzipped VCFs, and BCFs into genotype arrays.

    Args:
        vcf_filename: A str containing the name of a VCF, gzipped VCF, or BCF
            format file.
        ploidy: The ploidy of the data, 1 for phased data and 2 for unphased
            data.
        pass_str: If not None, the FILTER column of the VCF must match pass_str
             otherwise the SNP is ignored.  If None, all SNPs are included.

    Returns:
        A tuple of (genos, locs) as follows.
        genos: A num_loci by num_samples array containing the data with -1
               encoding missingness. genos only contains segregating sites.
        locs: A num_loci length array containing the genomic locations of the
              loci in genos.

    Raises:
        IOError: VCF must be for diploids.
        IOError: VCF must contain a single chromosome.
        IOError: SNP locations must be strictly increasing.
        IOError: Cannot have absent alleles for phased data.
    """
    logging.info('Reading data from %s', vcf_filename)
    genos = []
    locs = []
    chrom = None
    for variant in VCF(vcf_filename):
        if chrom and variant.CHROM != chrom:
            raise IOError('VCF must contain a single chromosome.')
        chrom = variant.CHROM
        if locs and locs[-1] == variant.start:
            continue
        if not variant.is_snp:
            if not (variant.REF == '0' and variant.ALT[0] == '1'):
                continue
        if len(variant.ALT) != 1:
            continue
        if pass_str and variant.FILTER != pass_str:
            continue
        gts = variant.genotype.array().astype(int)
        if gts.shape[1] != 3:
            raise IOError('VCF must be for diploids.')
        phased = gts[:, -1].astype(bool)
        gts = gts[:, :2]
        if ploidy == 2:
            gts[gts[:, 1] <= -1, 0] = -1
            gts[gts[:, 0] <= -1, 1] = -1
            gts = gts.sum(axis=1)
            gts[gts <= -1] = -1
        if ploidy == 1:
            gts[~phased, :] = np.minimum(gts[~phased, :], -1)
            gts = gts.flatten()
            if np.any(gts == -2):
                raise IOError('Cannot have absent alleles for phased data.')
        num_not_missing = (gts > -1).sum() * ploidy
        if num_not_missing == 0:
            continue
        alt_freq = gts[gts > -1].sum() / num_not_missing
        if alt_freq > 0 and alt_freq < 1:
            genos.append(gts)
            locs.append(variant.start)
    if len(genos) == 0:
        raise IOError('No valid SNPs found in VCF.')
    genos = np.array(genos)
    if np.any(np.diff(locs) <= 0.0):
        raise IOError('SNP locations must be strictly increasing.')
    logging.info('\tDone!')
    return genos, locs
