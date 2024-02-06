import numpy as np
from scipy.optimize import newton


def get_window_delimiters(
    number_of_bins_d, scaling_k, first_bin_size, embedding_step_size
):
    """
    Get delimiters of the window, used to describe the embedding. The
    window includes both the past embedding and the response.

    The delimiters are times, relative to the first bin, that separate
    two consequent bins.
    """

    bin_sizes = [
        first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
        for i in range(1, number_of_bins_d + 1)
    ]
    window_delimiters = [
        sum([bin_sizes[j] for j in range(i)]) for i in range(1, number_of_bins_d + 1)
    ]
    window_delimiters.append(
        window_delimiters[number_of_bins_d - 1] + embedding_step_size
    )
    return window_delimiters


def get_first_bin_size_for_embedding(embedding):
    """
    Get size of first bin for the embedding, based on the parameters
    T, d and k.
    """

    past_range_T, number_of_bins_d, scaling_k = embedding
    return newton(
        lambda first_bin_size: get_past_range(
            number_of_bins_d, first_bin_size, scaling_k
        )
        - past_range_T,
        0.005,
        tol=1e-03,
        maxiter=100,
    )


def get_past_range(number_of_bins_d, first_bin_size, scaling_k):
    """
    Get the past range T of the embedding, based on the parameters d, tau_1 and k.
    """

    return np.sum(
        [
            first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
            for i in range(1, number_of_bins_d + 1)
        ]
    )
