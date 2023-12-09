import pytest
import numpy as np
from scipy.stats import hypergeom
from scipy.stats import binom
from idtxl.postprocessing import SignificantSubgraphMining

# create sample data for tests of subgraph mining methods
# for between subjects design:
global network_samples_1
global network_samples_2
global SSM
network_samples_1 = [
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
]
network_samples_2 = [
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
]

SSM = SignificantSubgraphMining(
    network_samples_1, network_samples_2, design="between", alpha=0.05
)


# create sample data for tests of subgraph mining methods
# for between subjects design and idtxl format:
global network_samples_1_idtxl
global network_samples_2_idtxl
global SSM_idtxl
network_samples_1_idtxl = [
    [
        {"target": 0, "selected_vars_sources": []},
        {"target": 1, "selected_vars_sources": [(2, 5), (1, 5)]},
        {"target": 2, "selected_vars_sources": [(2, 5), (2, 10)]},
    ],
    [
        {"target": 0, "selected_vars_sources": [(1, 5)]},
        {"target": 1, "selected_vars_sources": [(0, 5)]},
        {"target": 2, "selected_vars_sources": [(2, 25)]},
    ],
]

network_samples_2_idtxl = [
    [
        {"target": 0, "selected_vars_sources": []},
        {"target": 1, "selected_vars_sources": [(2, 10), (1, 5)]},
        {"target": 2, "selected_vars_sources": [(2, 5), (2, 10)]},
    ],
    [
        {"target": 0, "selected_vars_sources": []},
        {"target": 1, "selected_vars_sources": [(0, 10), (1, 5)]},
        {"target": 2, "selected_vars_sources": [(0, 5)]},
    ],
]


SSM_idtxl = SignificantSubgraphMining(
    network_samples_1_idtxl,
    network_samples_2_idtxl,
    design="between",
    data_format="idtxl",
    alpha=0.5,
)


def test_idtxl_format():
    """test if coding_list and group networks generated correctly if idtxl
    data format is used"""

    correct_coding_list = [
        (1, 2, 10),
        (2, 0, 5),
        (2, 2, 25),
        (2, 2, 5),
        (1, 0, 10),
        (1, 1, 5),
        (0, 1, 5),
        (1, 2, 5),
        (2, 2, 10),
        (1, 0, 5),
    ]

    assert set(SSM_idtxl.coding_list) == set(correct_coding_list)
    assert len(SSM_idtxl.coding_list) == 10

    correct_groupA_networks = [[7, 5, 3, 8], [6, 9, 2]]
    correct_groupB_networks = [[0, 5, 3, 8], [4, 5, 1]]

    assert SSM_idtxl.groupA_networks == correct_groupA_networks
    assert SSM_idtxl.groupB_networks == correct_groupB_networks

    assert SSM_idtxl.count_subgraph([SSM_idtxl.coding_list.index((1, 1, 5))]) == (1, 2)
    assert SSM_idtxl.count_subgraph([SSM_idtxl.coding_list.index((2, 2, 5))]) == (1, 1)


# for within subject design:
global network_samples_1_within
global network_samples_2_within
global SSM_within
network_samples_1_within = [
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
]
network_samples_2_within = [
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
]

SSM_within = SignificantSubgraphMining(
    network_samples_1_within, network_samples_2_within, design="within", alpha=0.05
)


# example in which there are significant results
global network_samples_1_sign
global network_samples_2_sign
global SSM_sign
network_samples_1_sign = [
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
]
network_samples_2_sign = [
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1], [1, 0]]),
]

SSM_sign = SignificantSubgraphMining(
    network_samples_1_sign, network_samples_2_sign, design="between", alpha=0.05
)

SSM_sign_within = SignificantSubgraphMining(
    network_samples_1_sign, network_samples_2_sign, design="within", alpha=0.05
)

global permuted_data_set
permuted_data_set = [
    [SSM_sign.groupA_networks, SSM_sign.groupB_networks],
    [
        [[1], [0, 1], [1], [0, 1], [1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        [[0, 1], [1], [0, 1], [1], [0, 1], [1], [1], [1], [0, 1], [0, 1]],
    ],
]


def test_count_subgraph():
    """test count_subgraph function which counts the occurrences of a given
    subgraph in each group/condition"""

    assert SSM.count_subgraph([0]) == (7, 2)
    assert SSM.count_subgraph([1]) == (7, 5)
    assert SSM.count_subgraph([0, 1]) == (7, 2)


def test_encode_adjacency():
    """test encode_adjacency function that encodes all networks in the data
    as lists of link indices"""

    assert SSM.encode_adjacency() == (
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        [[1], [1], [1], [0, 1], [0, 1]],
    )


def test_decode_adjacency():
    """test decode_adjacency function that takes a list of link indices
    as input and returns corresponding adjacency matrix"""

    assert np.array_equal(SSM.decode_adjacency([0]), np.array([[0.0, 1.0], [0.0, 0.0]]))
    assert np.array_equal(SSM.decode_adjacency([1]), np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert np.array_equal(
        SSM.decode_adjacency([0, 1]), np.array([[0.0, 1.0], [1.0, 0.0]])
    )


def test_generate_min_p_table():
    """test generate_min_p_table method that generates a list of all
    minimal p-values (one per possible total number of occurrences)"""

    correct_min_p_values_between = [
        2.0,
        0.8333333333333334,
        0.30303030303030226,
        0.09090909090909093,
        0.020202020202020186,
        0.002525252525252523,
        0.01515151515151513,
        0.002525252525252523,
        0.02020202020202022,
        0.09090909090909093,
        0.30303030303030254,
        0.8333333333333334,
        2.0,
    ]

    # correct minimal p-values for within subject design assuming N=12
    correct_min_p_values_within = [2 * (1 / 2 ** i) for i in range(13)]

    assert all(
        np.isclose(SSM.generate_min_p_table("between"), correct_min_p_values_between)
    )

    assert all(
        np.isclose(SSM.generate_min_p_table("within"), correct_min_p_values_within)
    )


def test_count_discordants():
    """test count_discordants function which counts the numbers of the
    two types of discordant pairs: 1) subgraph occurs in condition A but not B
    2) subgraph occurs in condition B but not A."""

    assert SSM_within.count_discordants([0]) == (5, 0)
    assert SSM_within.count_discordants([1]) == (0, 0)
    assert SSM_within.count_discordants([0, 1]) == (5, 0)


def test_extend():
    """'test extend method that recursively enumerates all subgraphs
    occurring at least a given number of times."""

    # enumerate all subgraphs occuring at least 5 times. In the given data
    # set these are in fact all subgraphs because the fully connected graph
    # occurs more than 5 times.
    SSM.extend([], 5)
    assert SSM.frequent_graphs == [[0], [0, 1], [1]]

    # reset
    SSM.frequent_graphs = []
    SSM.minimum_p_values = []
    SSM.p_values = []

    # enumerate all subgraphs occuring at least 10 times. In the given data
    # set there is only one such graph.
    SSM.extend([], 10)
    assert SSM.frequent_graphs == [[1]]

    # reset
    SSM.frequent_graphs = []
    SSM.minimum_p_values = []
    SSM.p_values = []


def test_extend_wy():
    """'test extend_wy method that recursively determines the smallest
    observed p-value in the permuted data set"""

    # initialize permuted data set to original data set
    SSM.perm_groupA_networks = SSM.groupA_networks
    SSM.perm_groupB_networks = SSM.groupB_networks

    SSM.current_min_p = 10
    SSM.current_min_freq = 0

    SSM.extend_wy([])

    assert np.isclose(SSM.current_min_p, 0.09090909090909083)
    assert SSM.current_min_freq == 4


def test_extend_wy_mcnemar():
    """'test extend_wy method that recursively determines the smallest
    observed p-value in the permuted data set"""

    # initialize permuted data set to original data set
    SSM_within.perm_groupA_networks = SSM_within.groupA_networks
    SSM_within.perm_groupB_networks = SSM_within.groupB_networks

    SSM_within.current_min_p = 10
    SSM_within.current_min_freq = 0

    SSM_within.extend_wy_mcnemar([])

    assert np.isclose(SSM_within.current_min_p, 0.0625)
    assert SSM_within.current_min_freq == 5


def test_determine_tarone_factor():
    """'tests the determine_tarone_factor which computes the Tarone factor
    based on the calculated list of minimum p-values among testable graphs."""

    # construct an example were the Tarone factor should be 10
    SSM_sign.minimum_p_values = [0.05 / 20 for i in range(10)]
    SSM_sign.num_testable_graphs = 10
    SSM_sign.determine_tarone_factor()
    assert SSM_sign.k_rt == 10
    # reset
    SSM_sign.minimum_p_values = []

    # test by calling the enumerate_significant_subgraphs method which itself
    # calls the determine_tarone_factor method
    SSM_sign.enumerate_significant_subgraphs(method="Tarone")

    # all three subgraphs should be frequent given level 0.05
    assert SSM_sign.frequent_graphs == [[0], [0, 1], [1]]
    # only two subgraphs should be testable
    assert SSM_sign.num_testable_graphs == 2
    # correction factor should be 2
    assert SSM_sign.k_rt == 2


def test_westfall_young():
    """'test the westfall_young method that implements a permutation-by-
    permutation algorithm for the WY correction. The crucial functions
    extend_wy and extend_wy_mcnemar have already been tested above. Here
    are some additional tests regarding the permutation of the data set."""

    # For the between subjects case
    # is there a min p-value for each permuted data set?
    SSM_sign.westfall_young(num_perm=20)
    assert len(SSM_sign.permutation_min_p_values) == 20

    # are all graphs in the permuted data set actually in the original data set
    assert all(
        [
            SSM_sign.perm_groupA_networks[i] in SSM_sign.groupA_networks
            or SSM_sign.perm_groupA_networks[i] in SSM_sign.groupB_networks
            for i in range(10)
        ]
    )
    assert all(
        [
            SSM_sign.perm_groupA_networks[i] in SSM_sign.groupA_networks
            or SSM_sign.perm_groupA_networks[i] in SSM_sign.groupB_networks
            for i in range(10)
        ]
    )

    # Now the within subjects case...
    # is there a min p-value for each permuted data set?
    SSM_sign_within.westfall_young(num_perm=20)
    assert len(SSM_sign_within.permutation_min_p_values) == 20

    # are all graphs in the permuted data set actually in the original data set
    assert all(
        [
            SSM_sign_within.perm_groupA_networks[i] in SSM_sign_within.groupA_networks
            or SSM_sign_within.perm_groupA_networks[i]
            in SSM_sign_within.groupB_networks
            for i in range(10)
        ]
    )
    assert all(
        [
            SSM_sign_within.perm_groupA_networks[i] in SSM_sign_within.groupA_networks
            or SSM_sign_within.perm_groupA_networks[i]
            in SSM_sign_within.groupB_networks
            for i in range(10)
        ]
    )


def test_westfall_young_light():
    pass


def test_count_subgraph_wylight():
    """'test count_subgraph_wylight method that counts number of occurrences
    of a subgraph in k-th permuted data set"""

    SSM_sign.all_permuted_datasets = permuted_data_set
    assert SSM_sign.count_subgraph_wylight([0], 0) == (10, 2)
    assert SSM_sign.count_subgraph_wylight([0], 1) == (7, 5)
    assert SSM_sign.count_subgraph_wylight([1], 0) == (10, 10)
    assert SSM_sign.count_subgraph_wylight([1], 1) == (10, 10)
    assert SSM_sign.count_subgraph_wylight([0, 1], 0) == (10, 2)
    assert SSM_sign.count_subgraph_wylight([0, 1], 1) == (7, 5)


def test_count_discordants_wylight():
    """'test count_discordants_wylight method that counts number of discordant
    pairs of a subgraph in k-th permuted data set"""

    SSM_sign_within.all_permuted_datasets = permuted_data_set
    assert SSM_sign_within.count_discordants_wylight([0], 0) == (8, 0)
    assert SSM_sign_within.count_discordants_wylight([0], 1) == (5, 3)
    assert SSM_sign_within.count_discordants_wylight([1], 0) == (0, 0)
    assert SSM_sign_within.count_discordants_wylight([1], 1) == (0, 0)
    assert SSM_sign_within.count_discordants_wylight([0, 1], 0) == (8, 0)
    assert SSM_sign_within.count_discordants_wylight([0, 1], 1) == (5, 3)
