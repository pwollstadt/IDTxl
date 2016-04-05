# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:40:40 2016

@author: patricia
"""
import copy as cp
import numpy as np


def create_surrogates(realisations, replication_idx, n_perm, options=None):

    if options is None:
        options = {}
    try:
        perm_type = options['perm_type']
    except KeyError:
        perm_type = None

    # Check if there are enough data to create a sufficient number of
    # permutations.
    n_repl = np.unique(replication_idx).shape[0]
    samples_per_repl = sum(replication_idx == replication_idx[0])
    if (np.math.factorial(samples_per_repl) <= n_perm and
            np.math.factorial(n_repl) <= n_perm):
        raise RuntimeError('Number of samples per replication ({0}) and '
                           'number of replications ({1}) are each to small to '
                           'generate generate the requested number of '
                           'permutations ({2}).'.format(samples_per_repl,
                                                        n_repl, n_perm))

    if perm_type is None:
        # Check if n_repl is high enough to allow for the requested
        # number of permutations. If not permute samples over time
        if np.math.factorial(n_repl) > n_perm:
            permute_replications = True
        else:
            permute_replications = False
    elif (perm_type == 'permute_replications' and
          np.math.factorial(n_repl) <= n_perm):
        raise ValueError('Number of replications ({0}) is not high enough to '
                         'create the sufficient number of permutations for '
                         'surrogate testing.')
    elif (perm_type == 'permute_samples' and
          np.math.factorial(samples_per_repl) <= n_perm):
        raise ValueError('Number of samples per replications ({0}) is not high'
                         ' enough to create the sufficient number of '
                         'permutations for surrogate testing.')
    # TODO further checks go here as elifs

    if permute_replications:
        return permute_over_replications(realisations, replication_idx)
    else:
        return permute_over_time(realisations, replication_idx, options)


def permute_over_replications():
    return surrogates

def permute_over_time(realisations, replication_idx, perm_type, *kwargs):
    """Permute realisations in time within each replication.

    Permute realisations in time but within each replication. This is the
    fall-back option if the number of replications is too small to allow a
    sufficient number of permutations for the generation of surrogate data. If
    no permutation type is given, samples are randomly permuted over the
    whole replication, i.e., over all time indices in the replication.

    Args:
        realisations : numpy array
            shape[0] realisations of shape[1] variables
        replication_idx : numpy array
            index of replication a realisation came from
        perm_type : str
            type of permutation, can be 'random' (swaps samples at random),
            'blocks' (swaps blocks of samples), 'local' (swaps samples within
            a given range), or 'circular' (shifts time series by a random
            number of samples)
        *kwargs
            Arbitrary keyword arguments depending on the perm_type:
            if perm_type = 'blocks': 'block_size' in samples, 'swap_range' in
            blocks (default=max)
            if perm_type = 'local': 'perm_range' in samples, range over which
            realisations are permuted (default=max)
            if perm_type = 'circular': 'max_shift' in samples, the maximum
            number of samples for shifting (default=max)

    Returns:
        numpy array
            realisations permuted over time
    """
    assert (replication_idx.shape[0] == realisations.shape[0]), (
        'Array "replication" index must have as many entries as the first '
        'dimension of array "realisations".')

    samples_per_repl = sum(replication_idx == replication_idx[0])

    # Get the permutaion 'mask' for one replication (the same mask is then
    # applied to each replication).
    if perm_type == 'random':
        perm = np.random.permutation(n_per_repl)

    elif perm_type == 'blocks':
        try:
            block_size = kwargs['block_size']
        except KeyError:
            raise KeyError('No block size provided.')
        try:
            swap_range = kwargs['swap_range']
        except KeyError:
            swap_range = np.ceil(samples_per_repl / block_size).astype(int)
        perm = _swap_blocks(realisations, replication_idx,
                            block_size, swap_range)

    elif perm_type == 'local':
        try:
            perm_range = kwargs['perm_range']
        except KeyError:
            perm_range = samples_per_repl
        perm = _swap_local(samples_per_repl, perm_range)

    elif perm_type == 'circular':
        try:
            max_shift = kwargs['max_shift']
        except KeyError:
            max_shift = samples_per_repl - 1
        perm = _circular_shift(realisations, replication_idx, max_shift)

    else:
        raise ValueError('Unknown permutation type ({0}).'.format(perm_type))

    # Apply the permutation to data from each replication.
    realisations_perm = cp.copy(realisations)
    for replication in range(max(replication_idx) + 1):
        data_idx = replication_idx == replication
        d = realisations_perm[data_idx, :]
        realisations_perm[data_idx, :] = d[perm, :]

    return realisations_perm


def _swap_local(n_per_repl, perm_range):
    """Permute samples in a time series within a given range.

    If a permutation range is given, samples are shuffled within blocks of
    length permutation range.

    Args:
        n_per_repl : int
            number of samples in one replication
        perm_range : int
            range over which realisations are permuted

    Returns:
        numpy array
            permuted indices with length n_per_repl
    """
    assert (perm_range > 1), ('Permutation range has to be larger than 1',
                              'otherwise there is nothing to permute.')
    assert (n_per_repl >= perm_range), ('Not enough realisations per '
                                        'replication ({0}) to allow for the '
                                        'requested "perm_range" of {1}.'
                                        .format(n_per_repl, perm_range))

    # Create a permutation of the data that respects the requested permutation
    # range and can be applied to the realisations from each replication in
    # turn.
    if perm_range == n_per_repl:  # permute all realisations in one replication
        perm = np.random.permutation(n_per_repl)
    else:  # build a permutation that permutes only within the perm_range
        perm = np.empty(n_per_repl, dtype=int)
        remainder = n_per_repl % perm_range
        i = 0
        for p in range(n_per_repl // perm_range):
            perm[i:i + perm_range] = np.random.permutation(perm_range) + i
            i += perm_range
        if remainder > 0:
            perm[-remainder:] = np.random.permutation(remainder) + i
    return perm


def _swap_blocks(n_per_repl, block_size, swap_range):
    """Permute blocks of samples in a time series within a given range.

    Blocks of samples are permuted within a time series. If a swap range is
    given, blocks are only swapped with other blocks within that range.

    Args:
        n_per_repl : int
            number of samples in one replication
        block_size : int
            number of samples in a block
        swap_range : int
            range over which blocks can be swapped

    Returns:
        numpy array
            permuted indices with length n_per_repl
    """

    n_blocks = np.ceil(n_per_repl / block_size).astype(int)
    rem_samples = n_per_repl % block_size

    # First permute block(!) indices.
    if swap_range == n_blocks:  # permute all realisations in one replication
        perm = np.random.permutation(n_blocks)
    else:  # build a permutation that permutes only within the perm_range
        perm = np.empty(n_blocks, dtype=int)
        rem_blocks = n_blocks % swap_range
        i = 0
        for p in range(n_blocks // swap_range):
            perm[i:i + swap_range] = np.random.permutation(swap_range) + i
            i += swap_range
        if rem_blocks > 0:
            perm[-rem_blocks:] = np.random.permutation(rem_blocks) + i

    # Blow up block indices so we get one permuted index for each samples in a
    # block.
    return np.hstack((np.repeat(perm[:-1], block_size),
                      np.repeat(perm[-1:], rem_samples)))


def _circular_shift(n_per_repl, max_shift):
    """Permute a time series through shifting by a random number of samples.

    A time series is shifted circularly by a random number of samples. A
    circular shift of n means, that the last n samples are included at the
    beginning of the time series and all other sample indices are increased
    by n steps. max_shift is an upper limit for n.

    Args:
        n_per_repl : int
            number of samples in one replication
        max_shift: int
            maximum possible shift (default=n_per_repl)

    Returns:
        numpy array
            permuted indices with length n_per_repl
    """
    assert (max_shift < n_per_repl), ('Max_shift ({0}) has to be smaller than'
                                      'the number of samples in the time '
                                      'series ({1}).'.format(max_shift,
                                                             n_per_repl))
    shift = np.random.randint(max_shift + 1)

    return np.hstack((np.arange(n_per_repl - shift, n_per_repl),
                      np.arange(n_per_repl - shift)))
