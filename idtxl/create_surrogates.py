"""Create surrogate data.

Created on Tue Apr  5 16:40:40 2016

@author: patricia
"""
import copy as cp
import numpy as np
from . import idtxl_exceptions as ex

VERBOSE = True


def create_surrogates(realisations, replication_idx, n_perm, options=None):
    """Create appropriate surrogate data from realisations.

    Create surrogates by either permuting replications or samples in time,
    depending on the data structure. The function checks if the data contain
    enough replications to generate an appropriate amount of surrogate data
    given the requested number of permutations (e.g., to create 500 meaningful
    permutaions, we need at least 6 replications, because we can obtain
    6! = 720 permutations over replications). If data are not sufficient,
    surrogates are created by permuting data over time if the number of
    replications is not sufficient or permutation over time was explicitely
    requested in the options.

    Args:
        realisations : numpy array
            data to be permuted, where dimensions are
            realisation idx x variable
        replication_idx : numpy array
            each realisation's replication index, has the same dimension as
            realisation.shape[0]
        n_perm : int
            desired number of permutations the test for which surrogates are
            used
        options : dict [optional]
            options for surrogate creation, can contain:

            - 'perm_type' - 'permute_replications' for permutation of
            replications or 'permute_samples' for permutation of samples
            - 'shuffle_samples' - if permutation type is 'permute_samples'
            this defines the method used for permuting data over time, can be
            'random' (swaps samples at random), 'blocks' (swaps blocks of
            samples), 'local' (swaps samples within a given range), or
            'circular' (circular shift with given maximum)
            - depending on the shuffling method, further options may be
            defined, see help for function 'permute_over_time()' in this module
    """
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

    # If no permutation type was requested, try to generate surrogates by
    # permutation over replications.
    if perm_type is None:
        # Check if n_repl is high enough to allow for the requested number of
        # permutations. If not permute samples over time and warn the user
        # (if no. replications is low an explicit strategy for permuting
        # over time should be used, e.g. swapping blocks or circular shift).
        if np.math.factorial(n_repl) > n_perm:
            permute_replications = True
        else:
            ex.n_replications_low('The number of replications is too low to '
                                  'create the necessary number of permutaions,'
                                  ' consider specifying an alternative '
                                  'permutation scheme.')
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
    # Add further checks as elifs here if required.

    if permute_replications:
        return permute_replications(realisations, replication_idx)
    else:
        try:
            shuffle_samples = options['shuffle_samples']
        except KeyError:
            shuffle_samples = None
        return permute_samples(realisations, replication_idx, shuffle_samples,
                               options)


def permute_replications(realisations, replication_idx):
    """Permute replications while keeping temporal structure intact.

    Permute whole replications while keeping the temporal order of samples
    within single replications intact:

    original data:
        rep.:   1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5  6 6 6 6 ...
        sample: 1 2 3 4  1 2 3 4  1 2 3 4  1 2 3 4  1 2 3 4  1 2 3 4 ...

    permuted data:
        rep.:   3 3 3 3  1 1 1 1  4 4 4 4  6 6 6 6  2 2 2 2  5 5 5 5 ...
        sample: 1 2 3 4  1 2 3 4  1 2 3 4  1 2 3 4  1 2 3 4  1 2 3 4 ...

    Args:
        realisations : numpy array
            shape[0] realisations (over samples and replications) of
            shape[1] variables
        replication_idx : numpy array
            index of replication a realisation came from

    Returns:
        numpy array
            permuted realisations
        numpy array
            permuted indices of replications
    """
    samples_per_repl = sum(replication_idx == replication_idx[0])
    replications_perm = np.random.permutation(max(replication_idx))
    replication_idx_perm = np.repeat(replications_perm, samples_per_repl)
    realisations_perm = np.empty(realisations.shape)
    realisations_perm = realisations[replication_idx_perm, :]
    return realisations_perm, replication_idx_perm


def permute_samples(realisations, replication_idx=None, perm_opts=None):
    """Permute samples while keeping replications intact.

    Permute realisations in time while keeping the order of replications
    intact. Permutations can be performed under various restrictions:

    original data:
        rep.:   1 1 1 1 1 1 1 1  2 2 2 2 2 2 2 2  3 3 3 3 3 3 3 3 ...
        sample: 1 2 3 4 5 6 7 8  1 2 3 4 5 6 7 8  1 2 3 4 5 6 7 8 ...

    circular shift (default) by 2, 6, and 4 samples:
        rep.:   1 1 1 1 1 1 1 1  2 2 2 2 2 2 2 2  3 3 3 3 3 3 3 3  ...
        sample: 7 8 1 2 3 4 5 6  3 4 5 6 7 8 1 2  5 6 7 8 1 2 3 4  ...

    permute blocks of 3 samples:
        rep.:   1 1 1 1 1 1 1 1  2 2 2 2 2 2 2 2  3 3 3 3 3 3 3 3 ...
        sample: 4 5 6 7 8 1 2 3  1 2 3 7 8 4 5 6  7 8 4 5 6 1 2 3 ...

    permute data locally within a range of 4 samples:
        rep.:   1 1 1 1 1 1 1 1  2 2 2 2 2 2 2 2  3 3 3 3 3 3 3 3 ...
        sample: 1 2 4 3 8 5 6 7  4 1 2 3 5 7 8 6  3 1 2 4 8 5 6 7 ...

    random permutation:
        rep.:   1 1 1 1 1 1 1 1  2 2 2 2 2 2 2 2  3 3 3 3 3 3 3 3  ...
        sample: 4 2 5 7 1 3 2 6  7 5 3 4 2 1 8 5  1 2 4 3 6 8 7 5  ...

    Permuting samples is the fall-back option for surrogate creation if the
    number of replications is too small to allow for a sufficient number of
    permutations for the generation of surrogate data.

    Args:
        realisations : numpy array
            shape[0] realisations of shape[1] variables
        replication_idx : numpy array [optinal]
            grouping index to indicate the replication a realisation came from
        perm_opts : dict [optinal]
            options specifying the allowed permutations:

            - perm_type : str [optional]
              permutation type, can be

                - 'circular' (default): shifts time series by a random number
                  of samples
                - 'block': swaps blocks of samples,
                - 'local': swaps samples within a given range, or
                - 'random': swaps samples at random,

            - additional options depending on the perm_type (N is The
              number of samples re replication):

                - if perm_type == 'circular':
                  'max_shift': int
                      the maximum number of samples for shifting (default=N/2)
                - if perm_type == 'block':
                  'block_size' : int
                      no. samples per block (default=N/10)
                  'swap_range' : int
                      range in which blocks can be swapped (default=max)
                - if perm_type == 'local':
                  'perm_range' : int | str 'max'
                      range in samples over which realisations can be permuted,
                      also can be 'max' for the maximum number of samples,
                      (default=N/10)

    Returns:
        numpy array
            realisations permuted over time
        numpy Array
            permuted indices of samples
    """
    # Check if a replication index was provided, i.e., samples come from
    # different replications. If not, assume realisations come from the same
    # replication.
    if replication_idx is None:
        replication_idx = np.zeros(realisations.shape[0])
    else:
        assert (replication_idx.shape[0] == realisations.shape[0]), (
            'Array "replication" index must have as many entries as the first '
            'dimension of array "realisations".')

    samples_per_repl = sum(replication_idx == replication_idx[0])

    perm_type = perm_opts.get('perm_type', 'block')

    # Get the permutaion 'mask' for one replication (the same mask is then
    # applied to each replication).
    if perm_type == 'random':
        perm = np.random.permutation(samples_per_repl)

    elif perm_type == 'circular':
        max_shift = perm_opts.get('max_shift', round(samples_per_repl / 2))
        perm = _circular_shift(realisations, replication_idx, max_shift)[0]

    elif perm_type == 'block':
        block_size = perm_opts.get('block_size', round(samples_per_repl / 10))
        swap_range = perm_opts.get('swap_range', np.ceil(samples_per_repl /
                                                      block_size).astype(int))
        perm = _swap_blocks(realisations, replication_idx,
                            block_size, swap_range)

    elif perm_type == 'local':
        perm_range = perm_opts.get('perm_range', round(samples_per_repl / 10))
        if type(perm_range) is str:
            if perm_range != 'max':
                raise ValueError('Got {0} as input for perm_range. For '
                                 'permutation strategy ''local'' either an int'
                                 ' or ''max'' has to be provided.'.format(
                                                                  perm_range))
        perm = _swap_local(samples_per_repl, perm_range)

    else:
        raise ValueError('Unknown permutation type ({0}).'.format(perm_type))

    # Apply the permutation to data from each replication.
    realisations_perm = np.empty(realisations.shape)
    perm_idx = np.empty(realisations_perm.shape[0])
    for r in range(max(replication_idx) + 1):
        mask = replication_idx == r
        data_temp = realisations[mask, :]
        realisations_perm[mask, :] = data_temp[perm, :]
        perm_idx[mask] = perm

    return realisations_perm, perm_idx


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
    rem_blocks = n_blocks % swap_range
    if rem_samples == 0:
        rem_samples = block_size

    # First permute block(!) indices.
    if swap_range == n_blocks:  # permute all realisations in one replication
        perm = np.random.permutation(n_blocks)
    else:  # build a permutation that permutes only within the perm_range
        perm = np.empty(n_blocks, dtype=int)

        i = 0
        for p in range(n_blocks // swap_range):
            perm[i:i + swap_range] = np.random.permutation(swap_range) + i
            i += swap_range
        if rem_blocks > 0:
            perm[-rem_blocks:] = np.random.permutation(rem_blocks) + i

    idx_last_block = [i for i, j in enumerate(perm) if j == max(perm)][0]

    # Blow up block indices so we get one permuted index for each samples in a
    # block.
    return np.hstack((np.repeat(perm[:idx_last_block], block_size),
                      np.repeat(perm[idx_last_block], rem_samples),
                      np.repeat(perm[idx_last_block + 1:], block_size)))


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
        int
            no. samples by which the time series was shifted
    """
    assert (max_shift <= n_per_repl), ('Max_shift ({0}) has to be equal to or '
                                       ' smaller than the number of samples in'
                                       ' the time series ({1}).'.format(
                                                        max_shift, n_per_repl))
    shift = np.random.randint(low=1, high=max_shift + 1)
    if VERBOSE:
        print("replications are shifted by {0} samples".format(shift))

    return np.hstack((np.arange(n_per_repl - shift, n_per_repl),
                      np.arange(n_per_repl - shift))), shift
