import numpy as np
from scipy.stats import hypergeom
from scipy.stats import binom


class SignificantSubgraphMining:
    """Implementation of significant subgraph mining as described in

    Sugiyama M, Lopez FL, Kasenburg N, Borgwardt KM.  Significant
    subgraph mining with multiple testing correction.
    In:  Proceedings of the 2015 SIAMInternational Conference
    on Data Mining. SIAM; 2015. p. 37–45.

    Llinares-Lopez F, Sugiyama M, Papaxanthos L, Borgwardt K.  Fast and
    memory-efficient significant pattern mining via permutation testing.
    In:Proceedings of the 21th ACM SIGKDD International Conference on
    Knowledge Discovery and Data Mining. ACM; 2015. p. 725–734.

    Gutknecht, Wibral (2021): "Significant Subgraph Mining for Neural Network
    Inference with Multiple Comparisons Correction". bioRxiv.
    https://www.biorxiv.org/content/10.1101/2021.11.03.467050v1.full

    Attributes:
        resultsA : list
            List of lists of IDTxl results dicts. One list per subject in
            Group A and one results dict per target.
        resultsB : list
            List of lists of IDTxl results dicts. One list per subject in
            Group B and one results dict per target
        coding_list : list
            List of all target-source-lag triplets in data set. Used to encode
            networks as lists of indices.
        groupA_networks : list
            List of lists of indices representing networks of subjects in
            Group A
        groupB_networks : list
            List of lists of indices representing networks of subjects in
            Group B
        graph_type : string
            can be either "directed" or "undirected" (undirected is only
            possible if data_format is "adjacency")
        data_format : string
            can be either "idtxl" or "adjacency"
        design : string
            Sampling design. Either "within" for within-subject designs (the
            same group of subjects is measured under two different conditions)
            or "between" for between-subjects designs (two different groups
            of subjects are measured under the same condition)
        n_A : int
            Sample size of group A
        n_B : int
            Sample size of group B
        N : int
            Total sample size
        alpha : float
            Uncorrected significance level
        min_p_value_table : list
            List of minimum achievable p-values for all possible numbers of
            total occurrences of a subgraph (0 to N)
        p_value_table : numpy array
            Lookup table of p-values for each combination of occurrences in
            Group A and total occurrences between 0 and N
        min_freq : int
            Minimum number of occurrences required for testability at
            level alpha
        link_counts : list
            List of links counts for each link that occurs at least once
            in the data set (i.e. for each target-source-lag triplet
            in coding_list)
        union_indices : list
            List of indices of target-source-lag triplets in coding_list
            occuring at east min_freq times. All other triplets and all their
            supergraphs can be ignored because they are not even testable at
            level alpha
        frequent_graphs : list
            List of frequent subgraphs in data set occuring at least min_freq
            times. Initialized empty and filled by calling the
            enumerate_frequent_subgraphs method
        p_values : list
            List of p-values for each frequent subgraph. Initialized empty and
            filled by calling the enumerate_frequent_subgraphs method
        minimum_p_values : list
            List of minimum p-values for each frequent subgraph. Initialized
            empty and  filled by calling the enumerate_frequent_subgraphs
            method
        num_testable_graphs : int
            Number of subgraphs testable at level alpha. Initialized as 0 and
            determined by calling the enumerate_significant_subgraphs method.
        k_rt : int
            Tarones correction factor. Initialized as None. Determined by
            calling the enumerate_significant_subgraphs method
        p_values_corr : list
            List of corrected p-values. Intialized empty and filled by calling
            the enumerate_significant_subgraphs method
        significant_graphs : list
            List of tuples of significant subgraphs in data set and their
            associated corrected p-values. Initialized empty and filled by
            calling the enumerate_significant_subgraphs method
    """

    def __init__(
        self,
        resultsA,
        resultsB,
        alpha,
        design,
        graph_type="directed",
        data_format="adjacency",
    ):
        if not type(alpha) == float and not type(alpha) == int:
            raise TypeError("alpha must be of type float")

        # if not alpha < 1 and alpha > 0:
        #     raise ValueError("alpha must be strictly between 0 and 1")

        self.resultsA = resultsA
        self.resultsB = resultsB

        self.n_A = len(self.resultsA)
        self.n_B = len(self.resultsB)

        if design == "between":
            self.N = len(self.resultsA) + len(self.resultsB)
        elif design == "within":
            self.N = len(self.resultsA)

        self.design = design

        self.alpha = alpha

        self.graph_type = graph_type

        self.data_format = data_format

        self.min_p_value_table = self.generate_min_p_table(design)

        self.p_value_table = self.generate_p_table(design)

        try:
            self.min_freq = next(
                ind for ind in range(self.N + 1) if self.min_p_value_table[ind] <= alpha
            )
        except StopIteration:
            raise RuntimeError(
                f"Due to the sample size, a significant result at level {self.alpha} is in principle not possible."
            )

        # represent graphs as lists of link indices
        if data_format == "idtxl":
            self.coding_list = self.generate_coding_list()
            self.groupA_networks = self.encode()[0]
            self.groupB_networks = self.encode()[1]

        elif data_format == "adjacency":
            self.nodes = len(self.resultsA[0])
            self.coding_list = self.generate_coding_list()
            self.groupA_networks = self.encode_adjacency()[0]
            self.groupB_networks = self.encode_adjacency()[1]

        # count total occurrences for each link in the data set
        link_counts = []
        for i in range(len(self.coding_list)):
            link_counts.append(
                self.count_subgraph([i])[0] + self.count_subgraph([i])[1]
            )

        self.link_counts = link_counts

        self.union_indices = [
            i
            for i in range(len(self.link_counts))
            if self.link_counts[i] >= self.min_freq
        ]

        self.max_depth = np.infty

        # create ascending list of all possible p-values
        all_possible_p_values = set()
        for i in self.p_value_table:
            for j in i:
                all_possible_p_values.add(j)

        self.all_possible_p_values = sorted(list(all_possible_p_values), reverse=True)

        self.frequent_graphs = []
        self.p_values = []
        self.minimum_p_values = []
        self.k_rt = None
        self.p_values_corr = []
        self.num_testable_graphs = 0
        self.significant_subgraphs = []

    def generate_coding_list(self):
        """
        If data_format = "idtxl": Creates list of all target-source-lag
        triplets occuring at least once
        in the data set. This list is used to encode subject networks as
        lists including all indices of the coding list such that the
        corresponding triplet is part of the network.

        Returns:
            list of 3-tuples
                each tuple has the form (target index, source index, lags)


        If data_format = "adjacency": Creates list of all source-target tuples
        occuring at least once in the data set. This list is used to encode
        subject networks as lists including all indices of the coding list
        such that the corresponding tuple is part of the network.

        Returns:
            list of 2-tuples
                each tuple has the form (source index, target index)
        """
        if self.data_format == "idtxl":
            tsl_triplet_set = set()
            for group in [self.resultsA, self.resultsB]:
                for results_dicts_list in group:
                    for results_dict in results_dicts_list:
                        target = results_dict["target"]

                        for source in results_dict["selected_vars_sources"]:
                            tsl_triplet = (target, source[0], source[1])

                            tsl_triplet_set.add(tsl_triplet)

            tsl_triplet_list = list(tsl_triplet_set)

            return tsl_triplet_list

        elif self.data_format == "adjacency":
            st_tuple_set = set()

            if self.graph_type == "directed":
                # all possible source-target tuples
                tuples = [(i, j) for i in range(self.nodes) for j in range(self.nodes)]

            elif self.graph_type == "undirected":
                # only ascending source-target tuples
                tuples = [
                    (i, j)
                    for i in range(self.nodes)
                    for j in range(self.nodes)
                    if i < j
                ]

            # for each tuple, check if it occurs in the data set. If so, add
            # it to st_tuple_set
            for t in tuples:
                for adj in self.resultsA + self.resultsB:
                    if adj[t[0], t[1]] == 1:
                        st_tuple_set.add(t)

            return list(st_tuple_set)

    def encode(self):
        """
        Encodes all subject networks as lists of indices. The ith entry
        describes the occurrence of the ith target-source-lag triplet
        in the coding list (self.coding_list).

        Returns:
            tuple of lists of integers
                The first entry of the tuple is a list of integers
                describing the networks of subjects in Group A. The second
                entry is a list of integers for Group B.
        """

        all_networks = []
        for group in [self.resultsA, self.resultsB]:
            group_networks = []
            for results_dicts_list in group:
                subject_network = []

                for results_dict in results_dicts_list:
                    target = results_dict["target"]

                    for source in results_dict["selected_vars_sources"]:
                        subject_network.append(
                            self.coding_list.index((target, source[0], source[1]))
                        )

                group_networks.append(subject_network)
                0
            all_networks.append(group_networks)

        return all_networks[0], all_networks[1]

    def decode(self, indices):
        """
        Converts a given list of indices (representing a subgraph) into
        a list of corresponding target-source-lag triplets using the mapping
        described in the coding list.

        Args:
            indices : list of integers

        Returns:
            List of 3-tuples
        """

        tsl_triplets = []

        for i in indices:
            tsl_triplets.append(self.coding_list[i])

        return tsl_triplets

    def encode_adjacency(self):
        """Encodes all input adjacency matrices as lists of indices"""

        all_networks = []
        for group in [self.resultsA, self.resultsB]:
            group_networks = []
            for adj in group:
                subject_network = []
                for t in self.coding_list:
                    if adj[t[0], t[1]] == 1:
                        subject_network.append(self.coding_list.index(t))

                group_networks.append(subject_network)

            all_networks.append(group_networks)

        return all_networks[0], all_networks[1]

    def decode_adjacency(self, indices):
        """Decodes list of indices as adjacency matrix"""

        adjacency = np.zeros((self.nodes, self.nodes))

        for i in indices:
            t = self.coding_list[i]
            adjacency[t[0], t[1]] = 1

        return adjacency

    def generate_min_p_table(self, design):
        """
        Computes list of minimum p_values depending on the total number of
        occurrences and given the group sample sizes.

        Returns:
            list
                minimum p-values for each number of occurrences between 0 and N
        """

        if design == "between":
            min_p_value_table = []
            for m in range(self.N + 1):
                min_upper_left = np.max([0, m - self.n_B])
                max_upper_left = np.min([self.n_A, m])

                # consider most extreme cases

                # First option: put as many occurrences as possible in group A
                p_value_r = hypergeom.pmf(max_upper_left, M=self.N, n=m, N=self.n_A)

                # Second option: put as few occurrences as possible in group A
                p_value_l = hypergeom.pmf(min_upper_left, M=self.N, n=m, N=self.n_A)

                min_p_value_table.append(2 * np.min([p_value_r, p_value_l]))

            return min_p_value_table

        elif design == "within":
            min_p_value_table = []
            for d in range(self.N + 1):
                # consider most extreme cases

                # First option: put as many discordant pairs as possible in
                # condition A (meaning that they are all of the kind where
                # the subgraph occurred under condition A but not under
                # condition B)

                p_value_r = binom.pmf(d, d, 0.5)

                # Second option: put as few discordant pairs as possible in
                # condition A. Due to the symmetry of the Binomial distribution
                # this leads to the same p-value as first option.

                # p_value_l = binom.pmf(0, d, 0.5)

                min_p_value_table.append(2 * p_value_r)

            return min_p_value_table

    def generate_p_table(self, design):
        """
        Computes table of p-values depending on the total number of
        occurrences, the occurrences in Group A, and given the group
        sample sizes.

        Args:
            design : string
                sampling design. either "within" or "between"

        Returns:
            numpy array
                p-values for each number of occurrences and occurrences in
                Group A between 0 and N
        """

        if design == "between":
            p_value_table = np.zeros((self.N + 1, self.N + 1))
            for countA in range(self.N + 1):
                for occurrences in range(self.N + 1):
                    p_value_R = 1 - hypergeom.cdf(
                        countA - 1, M=self.N, n=occurrences, N=self.n_A
                    )

                    p_value_L = hypergeom.cdf(
                        countA, M=self.N, n=occurrences, N=self.n_A
                    )

                    p_value = 2 * np.min([p_value_R, p_value_L])

                    p_value_table[countA, occurrences] = p_value

            if self.n_A != self.n_B:
                return p_value_table

            else:
                # if sample sizes are equal, the numerical  representation
                # of p-values should be made
                # unique. Otherwise p-values for each
                # combination of countA and occurrences will be different due
                # to numerical errors even though they should be equal because
                # of symmetries. There are three such symmetries:

                # Firstly, the p-value for f(G) and f_1(G) is the same as that
                # for n - f(G) and (n_1 - f(G)) + f_1(G). For instance,
                # given a total sample sie of n= 40 we have that 10
                # total occurrences with three in group 1 is the same as
                # 30 total occurrences with 13 occurrences in group 1.
                # For this reason only p-values for f(G) up to floor(N/2) have
                # to be computed. p-values for f(G) equal to floor(N/2) + 1
                # up to N are always identical to one of the already computed
                # values

                p_value_table = np.zeros((self.N + 1, self.N + 1))
                for occurrences in range(int(self.N / 2) + 1):
                    for countA in range(occurrences + 1):
                        p_value_R = 1 - hypergeom.cdf(
                            countA - 1, M=self.N, n=occurrences, N=self.n_A
                        )

                        p_value_L = hypergeom.cdf(
                            countA, M=self.N, n=occurrences, N=self.n_A
                        )

                        p_value = 2 * np.min([p_value_R, p_value_L])

                        p_value_table[countA, occurrences] = p_value

                        p_value_table[
                            self.n_A - occurrences + countA, self.N - occurrences
                        ] = p_value

                # Secondly, given k total successes, the p-values corresponding to
                # k successes and 0 have to be the same. The same is true for
                # k-1 and 1, k-2 and 2, and so on.

                for occurrences in range(self.N + 1):
                    for countA in range(self.N + 1):
                        if countA < occurrences / 2:
                            p_value_table[countA, occurrences] = p_value_table[
                                occurrences - countA, occurrences
                            ]

                # Thirdly, if sample sizes are equal then each hypergeometric
                # distribution corresponding to a specific total number of
                # occurrences f(G) is symmetric. This means that if is uneven,
                # the p-values corresponding to f_1(G) = ceil(F(G) / 2) and
                # f_1(G) = floor(F(G) / 2) have to be equal to 1.

                for j in range(self.N + 1):
                    if j % 2 != 0:
                        p_value_table[int(np.floor(j / 2)), j] = 1
                        p_value_table[int(np.ceil(j / 2)), j] = 1

                return p_value_table

        elif design == "within":
            # compute p-value table based on Binomial distribution

            p_value_table = np.zeros((self.N + 1, self.N + 1))

            for d_in_A in range(self.N + 1):
                for d in range(self.N + 1):
                    p_value_r = 1 - binom.cdf(d_in_A - 1, d, 0.5)

                    p_value_l = binom.pmf(d_in_A, d, 0.5)

                    p_value_table[d_in_A, d] = 2 * np.min([p_value_r, p_value_l])

            return p_value_table

    # In the following two "count" methods and four "extend" methods are defined.

    # There is one extend method for each combination of sampling design
    # "within" / "between" and correction method "Tarone/Hommel" / "Westfall-Young"

    # Between + Tarone: extend()
    # Between + WY: extend_wy()
    # Within + Tarone: extend_mcnemar()
    # Within + WY: extend _wy_mcnemar()

    def count_subgraph(self, indices, where="original"):
        """
        Counts the number of occurrences of a subgraph represented by a list of
        indices

        Args:
            indices : list of integers
                indices of all links the subgraph to be counted consists of
            where : string
                if "original" then the subgraph is counted in the original
                data set. if "perm" the subgraph is counted in the permuted
                data set (for WY procedure)

        Returns:
            tuple of integers
                number of occurrences of subgraph in GroupA and in GroupB
        """

        if where == "original":
            graphsA = self.groupA_networks
            graphsB = self.groupB_networks
        elif where == "perm":
            graphsA = self.perm_groupA_networks
            graphsB = self.perm_groupB_networks

        countA = 0
        for graph in graphsA:
            # check if sub_graph occurs in graph
            if all(i in graph for i in indices):
                countA += 1

        countB = 0
        for graph in graphsB:
            # check if sub_graph occurs in graph
            if all(i in graph for i in indices):
                countB += 1

        return countA, countB

    def count_discordants(self, indices, where="original"):
        """
        Counts the discordant pairs for a given subgraph represented as a
        list of indices.

        Args:
            indices : list of integers
                indices of all links the subgraph to be counted consists of
            where : string
                if "original" then the discordants are counted in the original
                data set. if "perm" the discordants are counted in the permuted
                data set (for WY procedure)

        Returns:
            tuple of integers
                number of cases in which the subgraph occurred in condition A
                but not in B, and number of cases in which the subgraph
                occurred in B but not in A
        """

        if self.design == "between":
            raise RuntimeError(
                "The count_discordants method can only be used for within-subject designs. Currently the design is set to between subjects."
            )

        if where == "original":
            graphsA = self.groupA_networks
            graphsB = self.groupB_networks
        elif where == "perm":
            graphsA = self.perm_groupA_networks
            graphsB = self.perm_groupB_networks

        discordants_B = 0
        discordants_A = 0
        for ind in range(self.N):
            if all(i in graphsA[ind] for i in indices) and not all(
                i in graphsB[ind] for i in indices
            ):
                discordants_A += 1

            elif all(i in graphsB[ind] for i in indices) and not all(
                i in graphsA[ind] for i in indices
            ):
                discordants_B += 1

        return discordants_A, discordants_B

    def extend(self, to_be_extended, freq):
        """
        Recursively extends the input subgraph checking at each recursion
        step if the current subgraph occurs frequently enough to reach
        significance at level alpha. If this is not the case, it is not
        extended any further. If it is, the extend method is called again.
        Frequent subgraphs are appended to self.frequent_subgraphs.

        Args:
            to_be_extended : list
                list of indices describing the locations of 1s in the union
                network. Each such list represent a particular subgraph.
           freq : int
               desired minimum frequency
           max_depth : int
               If specified, only subgraphs with at most max_depth links are
               considered. For instance, if max_depth = 1 only individual
               links are tested. The default value is infinity meaning that
               all possible subgraphs are considered.

        Returns:
            None
        """

        # the variable "remaining" contains all union indices larger than the
        # largest index in to_be_extended. These are the indices considered
        # in the extension process. In this way the subgraphs to be extended
        # are always represented by an ascending list of indices making sure
        # that only one permutation of the same set of indices is checked
        # (e.g. only [0,1] and not also [1,0] since these represent exactly
        # the same subgraph)

        # the following if-else statement makes sure that the extend method
        # can also be called with an empty list of indices. In this case
        # all links occurring at least once in the data set are considered
        # for extension.

        if len(to_be_extended) < self.max_depth:
            if to_be_extended != []:
                max_index = np.max(to_be_extended)
            else:
                max_index = -1

            remaining = []
            for i in self.union_indices:
                if i > max_index:
                    remaining.append(i)

            # successively extend the current subgraph by the indices in
            # remaining
            for i in remaining:
                # create extended subgraph. to_be_extended is already in the
                # list of frequent subgraphs and should not be altered. Thus,
                # only a copy of it should be extended.
                new = to_be_extended.copy()
                new.append(i)

                # count occurrences of new subgraph in data set
                countA, countB = self.count_subgraph(new)

                occurrences = countA + countB

                # if the new subgraph occurs often enough, append it to
                # self.frequent_graphs and store its minimum and actual
                # p-value. Then apply the extend method to the new subgraph
                # again.
                if occurrences >= freq:
                    minimum_p = self.min_p_value_table[countA + countB]

                    self.minimum_p_values.append(minimum_p)

                    self.frequent_graphs.append(new)

                    p_value = self.p_value_table[countA, occurrences]

                    self.p_values.append(p_value)

                    self.extend(new, self.min_freq)

    def extend_mcnemar(self, to_be_extended, freq):
        """
        Same as extend() method but using McNemar's test for within subject
        designs

        Args:
            to_be_extended : list
                list of indices describing the locations of 1s in the union
                network. Each such list represent a particular subgraph.
           freq : int
               desired minimum frequency

        Returns:
            None
        """

        if len(to_be_extended) < self.max_depth:
            if to_be_extended != []:
                max_index = np.max(to_be_extended)
            else:
                max_index = -1

            remaining = []
            for i in self.union_indices:
                if i > max_index:
                    remaining.append(i)

            for i in remaining:
                new = to_be_extended.copy()
                new.append(i)

                countA, countB = self.count_subgraph(new)

                occurrences = countA + countB

                if occurrences >= freq:
                    # count discordant pairs
                    discordants_A, discordants_B = self.count_discordants(new)
                    discordants = discordants_A + discordants_B

                    # calculate and store p-value and minimal p-value
                    self.minimum_p_values.append(self.min_p_value_table[discordants])
                    p_value = self.p_value_table[discordants_A, discordants]
                    self.p_values.append(p_value)

                    self.frequent_graphs.append(new)
                    self.extend_mcnemar(new, self.min_freq)

    def extend_wy(self, to_be_extended):
        """
        Determines the smallest observed p-value in permuted version of the
        data set by recursively extending the input subgraph. At each
        recursion step the function checks if the current subgraph occurs
        frequently enough (> self.current_min_freq) to obtain a p-value smaller
        than the smallest p-value observed so far (self.current_min_p).
        If this is not the case, it is not extended any further. If it is,
        the actual p-value is calculated. If this p-value happens to be smaller
        than current_min_p, then current_min_p and self.current_min_freq are
        updated, and the extend method is called again. If this p-value happens
        to be larger than current_min_p, the extend method is called again
        immediately.

        Args:
            list of indices of coding list. Each such list represent
            a particular subgraph.
        Returns:
            None
        """
        if len(to_be_extended) < self.max_depth:
            if to_be_extended != []:
                max_index = np.max(to_be_extended)
            else:
                max_index = -1

            remaining = []
            for i in [j for j in range(len(self.coding_list))]:
                if i > max_index:
                    remaining.append(i)

            for i in remaining:
                new = to_be_extended.copy()
                new.append(i)

                countA, countB = self.count_subgraph(new, where="perm")

                occurrences = countA + countB

                if occurrences >= self.current_min_freq:
                    p_value = self.p_value_table[countA, occurrences]

                    if p_value < self.current_min_p:
                        # update current minimum p-value
                        self.current_min_p = p_value

                        # calculate minimum frequency necessary to obtain a p-value
                        # smaller than the current minimum observed p-value.

                        self.current_min_freq = next(
                            ind
                            for ind in range(self.N + 1)
                            if self.min_p_value_table[ind] <= self.current_min_p
                        )

                    self.extend_wy(new)

    def extend_wy_mcnemar(self, to_be_extended):
        """
        Same as extend_wy but using McNemars test

        Args:
            list of indices of coding list. Each such list represent
            a particular subgraph.
        Returns:
            None

        """

        if len(to_be_extended) < self.max_depth:
            if to_be_extended != []:
                max_index = np.max(to_be_extended)
            else:
                max_index = -1

            remaining = []
            for i in self.union_indices:
                if i > max_index:
                    remaining.append(i)

            for i in remaining:
                new = to_be_extended.copy()
                new.append(i)

                # count occurrences in permuted data set
                # the reasoning is that in order to obtain a certain
                # p-value, there must be a certain minimal number of
                # discordant pairs. But for this to be the case there
                # must be at this the same number of total occurrences.
                # Importantly, one cannot directly use the required number
                # of discordant pairs as a cut-off because a supergraph
                # may in fact have more discordant pairs than its subgraphs.

                countA, countB = self.count_subgraph(new, where="perm")

                occurrences = countA + countB

                if occurrences >= self.current_min_freq:
                    # count discordant pairs in permuted data set
                    discordants_A, discordants_B = self.count_discordants(
                        new, where="perm"
                    )
                    discordants = discordants_A + discordants_B

                    p_value = self.p_value_table[discordants_A, discordants]

                    if p_value < self.current_min_p:
                        # update current minimum p-value
                        self.current_min_p = p_value

                        # calculate minimum frequency necessary to obtain a p-value
                        # smaller than the current minimum observed p-value.

                        self.current_min_freq = next(
                            ind
                            for ind in range(self.N + 1)
                            if self.min_p_value_table[ind] <= self.current_min_p
                        )

                    self.extend_wy_mcnemar(new)

    def determine_tarone_factor(self):
        """
        Determines Tarone's correction factor in case there are at least two
        testable subgraphs.

        Returns:
            int
                Tarone's correction factor
        """
        if self.num_testable_graphs < 2:
            raise Exception(
                "There are less than two testable subgraphs. No correction required."
            )

        # Use bisection search to find Tarone's factor
        # the goal is to find the smallest integer k such that the ration of
        # the number of subgraphs testable at level alpha/k and k itself is
        # smaller than or equal to 1. We can restrict the search to the
        # interval 2 - number of alpha-testable subgraphs.

        criterion = False
        up = self.num_testable_graphs + 1
        low = 2
        max_iter = 1000
        iterations = 0
        while not criterion:
            iterations += 1

            if iterations == max_iter:
                print(
                    "WARNING: Correction factor could not be determined " + "after",
                    max_iter,
                    "iterations",
                )

            k_rt = int((up + low) / 2)

            # if the number of subgraphs testable at level alpha/k_rt is smaller
            # than k_rt, then the true k_rt must be smaller than or equal to
            # the current k_rt. So we set the current k_rt as the new upper
            # bound. Otherwise, the true k_rt must be larger, so we set the
            # current one as the new lower bound.
            if (
                np.sum(np.array(self.minimum_p_values) <= (self.alpha / k_rt)) - k_rt
                <= 0
            ):
                up = k_rt
            else:
                low = k_rt

            # We have found the true k_rt if:
            # the tarone criterion m(k) / k <= 1 is fulfilled for current k_rt
            # but not for the next smaller integer k_rt - 1
            criterion = (
                np.sum(np.array(self.minimum_p_values) <= (self.alpha / k_rt)) - k_rt
                <= 0
                and np.sum(np.array(self.minimum_p_values) <= (self.alpha / (k_rt - 1)))
                - k_rt
                + 1
                > 0
            )

        self.k_rt = k_rt

    def enumerate_frequent_graphs(self, freq):
        """
        Adds all subgraphs occuring at least freq times to self.frequent_graphs
        The process is carried out recursively using the extend() method.
        Individual links of the union network are successively extended to
        build more complex subgraphs. As soon as a subgraph does not occur
        often enough the extension process can be stopped because all
        supergraphs can at best occur with the same frequency. The extend()
        method also saves the minimum and actual p-values of all frequent
        subgraphs.

        Args:
            freq : int
                desired minimum frequency
        """
        # create union network consisting of all individual links occuring
        # at least freq times
        self.union_indices = [
            i for i in range(len(self.link_counts)) if self.link_counts[i] >= freq
        ]

        # reinitialize set of frequent subgraphs
        self.frequent_graphs = []

        # reinititalize list of actual p-values
        self.p_values = []

        # reinitialize list of minimum p-values
        self.minimum_p_values = []

        # recursively extend subgraphs starting from individual links in the
        # union network
        if self.design == "between":
            self.extend([], freq)

            return self.frequent_graphs

        elif self.design == "within":
            self.extend_mcnemar([], freq)

            return self.frequent_graphs

    def enumerate_significant_subgraphs(
        self,
        method="Hommel",
        wy_algorithm="simple_depth_first",
        verbose=True,
        num_perm=10000,
        max_depth=np.infty,
    ):
        """
        This is the main function carrying out significant subgraph mining
        according to the multiple comparisons correction method (and algorithm
        in the case of Westfall-Young) of choice. It calls the relevant
        methods depending on the input arguments.

        Args:
            verbose : bool
                If True, print summary of results
            method : string
                Determines method used for multiple comparisons correction. can
                be "Tarone", "Hommel", or "Westfall-Young"
            num_perm : int
                Number of permutations used for Westfall-Young procedure.
            wy_algorithm : string
                algorithm used for Westfall-Young permutation procedure. Can be
                either "simple_depth_fist" (evaluates one permuted data set
                at a time) or "wy_light" for the Westfall-Young light algorithm
                introduced by Llinares-Lopez et al 2015 (distributes
                computations across permutations)
            max_depth : integer
                maximum complexity of subgraphs (number of links) up to which
                subgraphs are mined.

        Returns:
            list of tuples
                The first entry of each tuple is a list of indices representing
                the identified significant subgraph. The second entry is the
                associated (uncorrected) p-value.
        """

        self.max_depth = max_depth

        if method != "Hommel" and method != "Tarone" and method != "Westfall-Young":
            raise Exception("Method must be 'Hommel' or 'Tarone' or 'Westfall-Young'")

        if verbose is True:
            if self.max_depth == np.infty:
                print("Search Space: All possible subgraphs")
            else:
                print(
                    "Search Space: Subgraphs consisting of up to",
                    self.max_depth,
                    "links",
                )

            if self.design == "between":
                print("Between-Subjects Design: Using Fishers Exact Test")
                print()
            elif self.design == "within":
                print("Within-Subject Design: Using McNemars Test")
                print()

        # if union network is empty, there are no testable subgraphs and hence
        # no significant subgraphs
        if self.union_indices == []:
            print(
                "There are no alpha-testable subgraphs. No significant differences detectable."
            )
            return None

        # reinitialize list of significant subgraphs
        self.significant_subgraphs = []

        if method == "Westfall-Young":
            if wy_algorithm == "simple_depth_first":
                self.westfall_young(num_perm, verbose)
                return self.significant_subgraphs

            elif wy_algorithm == "wy_light":
                self.westfall_young_light(num_perm, verbose)
                return self.significant_subgraphs

            # elif wy_algorithm == "wy_light" and self.design == "within":
            #     raise Exception("wy_light algorithm is currently only "
            #             "implemented for between subjects designs")

            else:
                raise Exception(
                    "wy_algorithm has to be either simple_depth_first or wy_light"
                )

        # if method is not Westfall-Young the set of frequent subgraphs has
        # to be determined (all subgraphs occuring often enough to reach
        # significance at level alpha)
        if verbose is True:
            print("Determining frequent subgraphs...")

        self.enumerate_frequent_graphs(self.min_freq)

        # If there are no testable subgraphs at level alpha, there can be no
        # significant subgraphs
        # count alpha testable graphs
        self.num_testable_graphs = np.sum(np.array(self.minimum_p_values) <= self.alpha)

        if verbose is True:
            print("Number of frequent subgraphs: ", len(self.frequent_graphs))
            print("Number of alpha-testable subgraphs:", self.num_testable_graphs)
            print()

        if self.num_testable_graphs == 0:
            print(
                "There are no testable subgraphs. No significant "
                + "differences detectable."
            )
            return None

        elif self.num_testable_graphs == 1:
            self.k_rt = 1

        else:
            # determine Tarone's correction factor
            if verbose is True:
                print("Determining Tarone correction factor...")

            self.determine_tarone_factor()

        if method == "Hommel":
            hommel_level = sorted(self.minimum_p_values)[int(self.k_rt - 1)]
            corrected_level = np.max([hommel_level, (self.alpha / self.k_rt)])

            if verbose is True:
                print("Using Hommel correction...")
                print("Corrected level:", corrected_level)
                print("Correction factor:", self.alpha / corrected_level)
                # print("Hommel level", hommel_level)
                print()

            # determine significant subgraphs
            run = 0
            for p in self.p_values:
                if p <= self.alpha / self.k_rt or p < hommel_level:
                    self.significant_subgraphs.append(
                        (
                            self.decode(self.frequent_graphs[run]),
                            self.p_values[run] * (self.alpha / corrected_level),
                        )
                    )

                run += 1

            if verbose is True:
                print(
                    len(self.significant_subgraphs), "significant subgraphs identified."
                )

            return self.significant_subgraphs

        elif method == "Tarone":
            if verbose is True:
                print("Correction factor:", self.k_rt)
                print("Corrected level:", self.alpha / self.k_rt)
                print()
                print("Using Tarone correction...")

            # correct p-values
            self.p_values_corr = np.array(self.p_values)
            self.p_values_corr = self.k_rt * self.p_values_corr

            # determine significant subgraphs and corresponding p-values
            for i in range(len(self.p_values)):
                if self.p_values_corr[i] <= self.alpha:
                    self.significant_subgraphs.append(
                        (self.decode(self.frequent_graphs[i]), self.p_values[i])
                    )

            if verbose is True:
                print(
                    len(self.significant_subgraphs), "significant subgraphs identified."
                )

            return self.significant_subgraphs

    def westfall_young(self, num_perm=10000, verbose=True):
        """
        Determines significant subgraphs using the Westfall-Young Permutation
        procedure for multiple comparisons correction. This algorithm computes
        the permutation distribution of the smallest observed p-value
        permutation-by-permutation.

        Args:
            num_perm : int
                Number of permutation used for Westfall-Young procedure.
            verbose : bool
                If True, print summary of results

        Returns:
            None
        """
        if verbose is True:
            print(
                "Determining permutation distribution based on",
                num_perm,
                "permutations...",
            )

        # INITIALIZATION

        # current_min_p is the smallest p-value observed so far. Should be
        # initialized at some value larger than the largest possible p-value
        # (which in this case is 2)
        self.current_min_p = 10

        # minimum frequency necessary to reach a p-value smaller than
        # current_min_p
        self.current_min_freq = 0

        # permutated versions of the data set. initialized at the original data
        # set
        self.perm_groupA_networks = self.groupA_networks.copy()
        self.perm_groupB_networks = self.groupB_networks.copy()

        # list to store minimum observed p-values for each permutation
        self.permutation_min_p_values = []

        # initialize class labels
        group_idx = np.zeros(self.N)
        group_idx[self.n_B :] = 1

        all_networks = self.groupA_networks + self.groupB_networks

        # END OF INITIALIZATION

        # calculate minimum observed p-value for original data set
        if self.design == "between":
            self.extend_wy([])
        elif self.design == "within":
            self.extend_wy_mcnemar([])

        self.permutation_min_p_values.append(self.current_min_p)

        # reinitialize current_min_p
        self.current_min_p = 10

        # find minimum observed p-values for num_perm permutated data sets
        for i in range(num_perm - 1):
            # create empty network lists
            self.perm_groupA_networks = []
            self.perm_groupB_networks = []

            # generate permuted class labels and add networks to lists
            if self.design == "between":
                perm = np.random.permutation(group_idx)

                for i in range(len(perm)):
                    if perm[i] == 0:
                        self.perm_groupA_networks.append(all_networks[i])
                    else:
                        self.perm_groupB_networks.append(all_networks[i])

                # calculate minimum observed p-value for permuted data set
                self.extend_wy([])
                self.permutation_min_p_values.append(self.current_min_p)

            elif self.design == "within":
                # decide independently for each subject if outcome is flipped
                perm = np.random.choice([1, 0], size=self.N)

                for i in range(len(perm)):
                    if perm[i] == 0:
                        self.perm_groupA_networks.append(self.groupA_networks[i])
                        self.perm_groupB_networks.append(self.groupB_networks[i])
                    else:
                        self.perm_groupA_networks.append(self.groupB_networks[i])
                        self.perm_groupB_networks.append(self.groupA_networks[i])

                # calculate minimum observed p-value for permuted data set
                self.extend_wy_mcnemar([])
                self.permutation_min_p_values.append(self.current_min_p)

            # reinitialize current_min_p and current_min_freq
            self.current_min_p = 10
            self.current_min_freq = 0

        # calculate corrected level as largest value delta in
        # permutation_min_p_values such that the fraction of values in
        # permutation_min_p_values smaller or equal to delta is smaller or
        # equal to self.alpha
        delta = None

        # The maximum, i.e. the correction factor, should be determined
        # over the set of all possible p-values. This is because choosing
        # the correction factor between two possible p-values p_1 and p_2
        # leads to the same results as simply setting it to p_1. Hence we only
        # have to maximize over all possible p-values instead of over the
        # entire real interval (0,alpha) using grid search.

        for i in sorted(self.all_possible_p_values):
            # count fraction of minimum p values below ith entry
            if (
                np.sum(np.array(self.permutation_min_p_values) <= i) / num_perm
            ) <= self.alpha:
                delta = i
            else:
                break

        if delta is not None:
            self.wy_corrected_level = delta
        else:
            self.wy_corrected_level = 0
            print("Corrected level is zero. Try a larger number of permutations.")
            return None

        # The final step is to determine which subgraphs reach significant at
        # the corrected level. In order to do so it is not necessary to compute
        # the p-values of all subgraphs but only those occuring often enough
        # to reach significance at the corrected level:

        self.min_freq_wy = next(
            ind
            for ind in range(self.N + 1)
            if self.min_p_value_table[ind] <= self.wy_corrected_level
        )

        self.enumerate_frequent_graphs(self.min_freq_wy)

        self.significant_subgraphs = []
        for i in range(len(self.p_values)):
            if self.p_values[i] <= self.wy_corrected_level:
                self.significant_subgraphs.append(
                    (
                        self.decode(self.frequent_graphs[i]),
                        self.p_values[i] * (self.alpha / self.wy_corrected_level),
                    )
                )

        if verbose is True:
            print("Corrected level:", self.wy_corrected_level)
            print(len(self.significant_subgraphs), "significant subgraphs identified.")

    def westfall_young_light(self, num_perm=10000, verbose=True):
        """
        Determines significant subgraphs using the Westfall-Young light
        algorithm described in

        Llinares-Lopez F, Sugiyama M, Papaxanthos L, Borgwardt K.  Fast and
        memory-efficient significant pattern mining via permutation testing.
        In:Proceedings of the 21th ACM SIGKDD International Conference on
        Knowledge Discovery and Data Mining. ACM; 2015. p. 725–734.

        Args:
            num_perm : int
                Number of permutation used for Westfall-Young procedure.
            verbose : bool
                If True, print summary of results

        Returns:
            None
        """
        if verbose is True:
            print(
                "Determining permutation distribution based on",
                num_perm,
                "permutations...",
            )
            print("Design:", self.design)
        self.current_min_freq = 0

        self.num_perm = num_perm

        # initialize Westfall-Young corrected significance level
        self.wy_level_light = 1

        # initialize list of smallest p-values for each permuted data set
        self.smallest_p_perm = np.ones(num_perm)

        # initialize permuted data sets
        self.all_permuted_datasets = []

        # vector to be permuted
        group_idx = np.zeros(self.N)
        group_idx[self.n_B :] = 1

        all_networks = self.groupA_networks + self.groupB_networks

        # add original dataset as first entry to all_permuted_datasets
        self.all_permuted_datasets.append([])
        self.all_permuted_datasets[-1].append(self.groupA_networks)
        self.all_permuted_datasets[-1].append(self.groupB_networks)

        if self.design == "between":
            # add permutations and store all permuted data sets
            for i in range(1, num_perm):
                # create random permutation of class labels
                perm = np.random.permutation(group_idx)

                # add corresponding data set to list of all permuted data sets
                self.all_permuted_datasets.append([[], []])

                for i in range(self.N):
                    if perm[i] == 0:
                        self.all_permuted_datasets[-1][0].append(all_networks[i])
                    else:
                        self.all_permuted_datasets[-1][1].append(all_networks[i])

        elif self.design == "within":
            # add permutations and store all permuted data sets
            for i in range(1, num_perm):
                # decide independently for each subject if outcome is flipped
                perm = np.random.choice([1, 0], size=self.N)

                # add corresponding data set to list of all permuted data sets
                self.all_permuted_datasets.append([[], []])

                for i in range(self.N):
                    if perm[i] == 0:
                        self.all_permuted_datasets[-1][0].append(
                            self.groupA_networks[i]
                        )
                        self.all_permuted_datasets[-1][1].append(
                            self.groupB_networks[i]
                        )

                    else:
                        self.all_permuted_datasets[-1][0].append(
                            self.groupB_networks[i]
                        )
                        self.all_permuted_datasets[-1][1].append(
                            self.groupA_networks[i]
                        )

        # start extension process
        if self.design == "between":
            self.extend_wy_light([])
        elif self.design == "within":
            self.extend_wy_light_mcnemar([])

        # index of current estimate of WY threshold
        wy_index = self.all_possible_p_values.index(self.wy_level_light)
        # look for final threshold within set of possible p-values up to the
        # current estimate (sanity check)
        search_in = self.all_possible_p_values[wy_index:]

        ind = 0
        while (
            np.sum(self.smallest_p_perm <= search_in[ind]) / self.num_perm > self.alpha
        ):
            ind += 1

        self.wy_level_light = search_in[ind]

        if verbose is True:
            print("WY-level determined...")

        self.min_freq_wy = next(
            ind
            for ind in range(self.N + 1)
            if self.min_p_value_table[ind] <= self.wy_level_light
        )

        self.enumerate_frequent_graphs(self.min_freq_wy)

        self.significant_subgraphs = []
        for i in range(len(self.p_values)):
            if self.p_values[i] <= self.wy_level_light:
                self.significant_subgraphs.append(
                    (
                        self.decode(self.frequent_graphs[i]),
                        self.p_values[i] * (self.alpha / self.wy_level_light),
                    )
                )

        if verbose is True:
            print("Corrected level:", self.wy_level_light)
            print(len(self.significant_subgraphs), "significant subgraphs identified.")

    def extend_wy_light(self, to_be_extended):
        """Westfall-Young light extension method. Evaluates all permutations
        at the same time for each subgraph. The goal is to determine the
        Westfall-Young corrected level, i.e. the alpha quantile of the
        permutation distribution of the smallest observed p-value among
        subgraphs.  Recursively, evaluates subgraphs and updates
        the current estimate of the Westfall-Young corrected level
        self.wy_level_light.

        Args:
            indices : list of integers
                indices of all links of the subgraph
        Returns:
            None
        """

        if len(to_be_extended) < self.max_depth:
            if to_be_extended != []:
                max_index = np.max(to_be_extended)
            else:
                max_index = -1

            remaining = []
            for i in [j for j in range(len(self.coding_list))]:
                if i > max_index:
                    remaining.append(i)

            for i in remaining:
                new = to_be_extended.copy()
                new.append(i)

                # calculate minimum achievable p-value
                countA, countB = self.count_subgraph(new)
                occurrences = countA + countB
                min_p = self.min_p_value_table[occurrences]

                # only continue if subgraph if testable at current estimate
                # of wy_level
                if min_p <= self.wy_level_light:
                    # calculate p-value for all permuted data sets:
                    for k in range(self.num_perm):
                        countA_perm, countB_perm = self.count_subgraph_wylight(new, k)
                        p_value = self.p_value_table[countA_perm, occurrences]

                        # update smallest p-value for k-th permuted data set
                        self.smallest_p_perm[k] = min(
                            [p_value, self.smallest_p_perm[k]]
                        )

                    # estimate of FWER
                    FWER = (
                        np.sum(self.smallest_p_perm <= self.wy_level_light)
                        / self.num_perm
                    )

                    while FWER > self.alpha:
                        # decrease corrected level while FWER is greater than alpha
                        self.wy_level_light = next(
                            p
                            for p in self.all_possible_p_values
                            if p < self.wy_level_light
                        )
                        FWER = (
                            np.sum(self.smallest_p_perm <= self.wy_level_light)
                            / self.num_perm
                        )

                # check if subgraph occurs often enough to be testable
                # at updated significance level
                # Only if this is the case, extend it further

                self.current_min_freq = next(
                    ind
                    for ind in range(self.N + 1)
                    if self.min_p_value_table[ind] <= self.wy_level_light
                )

                if occurrences >= self.current_min_freq:
                    self.extend_wy_light(new)

    def count_subgraph_wylight(self, indices, k):
        """
        Counts subgraph occurrences in k-th permuted data set

        Args:
            indices : list of integers
                indices of all links of the subgraph
            k : integer
                index of permuted data set

        Returns:
            tuple of integers
                number of occurrences in group A and number of occurrences
                in group B
        """
        countA = 0
        for graph in self.all_permuted_datasets[k][0]:
            # check if sub_graph occurs in graph
            if all(i in graph for i in indices):
                countA += 1

        countB = 0
        for graph in self.all_permuted_datasets[k][1]:
            # check if sub_graph occurs in graph
            if all(i in graph for i in indices):
                countB += 1

        return countA, countB

    def extend_wy_light_mcnemar(self, to_be_extended):
        """Westfall-Young light extension method for the within-subjects case
        using McNemars test. Recursively, evaluates subgraphs and updates
        the current estimate of the Westfall-Young corrected level
        self.wy_level_light.

        Args:
            indices : list of integers
                indices of all links of the subgraph
        Returns:
            None
        """

        if len(to_be_extended) < self.max_depth:
            if to_be_extended != []:
                max_index = np.max(to_be_extended)
            else:
                max_index = -1

            remaining = []
            for i in [j for j in range(len(self.coding_list))]:
                if i > max_index:
                    remaining.append(i)

            for i in remaining:
                new = to_be_extended.copy()
                new.append(i)

                # count occurrences
                countA, countB = self.count_subgraph(new)
                occurrences = countA + countB

                # calculate minimum achievable p-value
                discordants_A, discordants_B = self.count_discordants(new)
                discordants = discordants_A + discordants_B
                min_p = self.min_p_value_table[discordants]

                # only continue if subgraph if testable at current estimate
                # of wy_level
                if min_p <= self.wy_level_light:
                    # calculate p-value for all permuted data sets:
                    for k in range(self.num_perm):
                        (
                            discordants_A_perm,
                            discordants_B_perm,
                        ) = self.count_discordants_wylight(new, k)
                        p_value = self.p_value_table[discordants_A_perm, discordants]

                        # update smallest p-value for k-th permuted data set
                        self.smallest_p_perm[k] = min(
                            [p_value, self.smallest_p_perm[k]]
                        )

                    # estimate of FWER
                    FWER = (
                        np.sum(self.smallest_p_perm <= self.wy_level_light)
                        / self.num_perm
                    )

                    while FWER > self.alpha:
                        # decrease corrected level while FWER is greater than alpha
                        self.wy_level_light = next(
                            p
                            for p in self.all_possible_p_values
                            if p < self.wy_level_light
                        )
                        FWER = (
                            np.sum(self.smallest_p_perm <= self.wy_level_light)
                            / self.num_perm
                        )

                # check how many discordant pairs would be needed to obtain
                # a p-value significant at the new wy_level

                self.current_min_freq = next(
                    ind
                    for ind in range(self.N + 1)
                    if self.min_p_value_table[ind] <= self.wy_level_light
                )

                # Only if there are enough occurrences to obtain this number
                # of discordant pairs extend the graph further
                if occurrences >= self.current_min_freq:
                    self.extend_wy_light_mcnemar(new)

    def count_discordants_wylight(self, indices, k):
        """
        Counts discordant pairs for subgraph given by list if indices
        in k-th permuted data set.

        Args:
            indices : list of integers
                indices of all links of the subgraph
            k : integer
                index of permuted data set

        Returns:
            tuple of integers
                number of cases in which the subgraph occurred in condition A
                but not in B, and number of cases in which the subgraph
                occurred in B but not in A

        """

        graphsA = self.all_permuted_datasets[k][0]
        graphsB = self.all_permuted_datasets[k][1]

        discordants_B = 0
        discordants_A = 0
        for ind in range(self.N):
            if all(i in graphsA[ind] for i in indices) and not all(
                i in graphsB[ind] for i in indices
            ):
                discordants_A += 1

            elif all(i in graphsB[ind] for i in indices) and not all(
                i in graphsA[ind] for i in indices
            ):
                discordants_B += 1

        return discordants_A, discordants_B
