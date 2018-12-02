"""Provide results class for IDTxl network analysis."""
import sys
import warnings
import copy as cp
import numpy as np
from . import idtxl_utils as utils

warnings.simplefilter(action='ignore', category=FutureWarning)
MIN_INT = -sys.maxsize - 1  # minimum integer for initializing adj. matrix


class DotDict(dict):
    """Dictionary with dot-notation access to values.

    Provides the same functionality as a regular dict, but also allows
    accessing values using dot-notation.

    Example:

        >>> from idtxl.results import DotDict
        >>> d = DotDict({'a': 1, 'b': 2})
        >>> d.a
        >>> # Out: 1
        >>> d['a']
        >>> # Out: 1
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        """Return dictionary keys as list of attributes."""
        return self.keys()

    def __deepcopy__(self, memo):
        """Provide deep copy capabilities.

        Following a fix described here:
        https://github.com/aparo/pyes/pull/115/commits/d2076b385c38d6d00cebfe0df7b0d1ba8df934bc
        """
        dot_dict_copy = DotDict([
            (cp.deepcopy(k, memo),
             cp.deepcopy(v, memo)) for k, v in self.items()])
        return dot_dict_copy

    def __getstate__(self):
        # For pickling the object
        return self

    def __setstate__(self, state):
        # For un-pickling the object
        self.update(state)
        # self.__dict__ = self


class AdjacencyMatrix():
    """Adjacency matrix representing inferred networks."""
    def __init__(self, n_nodes, weight_type):
        self._edge_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
        self._weight_matrix = np.zeros((n_nodes, n_nodes), dtype=weight_type)
        if np.issubdtype(weight_type, np.integer):
            self._weight_type = np.integer
        elif np.issubdtype(weight_type, np.float):
            self._weight_type = np.float
        elif weight_type is bool:
            self._weight_type = weight_type
        else:
            raise RuntimeError('Unknown weight data type {0}.'.format(
                weight_type))

    def n_nodes(self):
        """Return number of nodes."""
        return self._edge_matrix.shape[0]

    def n_edges(self):
        return self._edge_matrix.sum()

    def add_edge(self, i, j, weight):
        """Add weighted edge (i, j) to adjacency matrix."""
        if not np.issubdtype(type(weight), self._weight_type):
            raise TypeError(
                'Can not add weight of type {0} to adjacency matrix of type '
                '{1}.'.format(type(weight), self._weight_type))
        self._edge_matrix[i, j] = True
        self._weight_matrix[i, j] = weight

    def add_edge_list(self, i_list, j_list, weights):
        """Add multiple weighted edges (i, j) to adjacency matrix."""
        if len(i_list) != len(j_list):
            raise RuntimeError(
                'Lists with edge indices must be of same length.')
        if len(i_list) != len(weights):
            raise RuntimeError(
                'Edge weights must have same length as edge indices.')
        for i, j, weight in zip(i_list, j_list, weights):
            self.add_edge(i, j, weight)

    def print_matrix(self):
        """Print weight and edge matrix."""
        print(self._edge_matrix)
        print(self._weight_matrix)

    def get_edge_list(self):
        """Return list of weighted edges.

        Returns
            list of tuples
                each entry represents one edge in the graph: (i, j, weight)
        """
        edge_list = np.zeros(self.n_edges(), dtype=object)  # list of tuples
        ind = 0
        for i in range(self.n_nodes()):
            for j in range(self.n_nodes()):
                if self._edge_matrix[i, j]:
                    edge_list[ind] = (i, j, self._weight_matrix[i, j])
                    ind += 1
        return edge_list


class Results():
    """Parent class for results of network analysis algorithms.

    Provide a container for results of network analysis algorithms, e.g.,
    MultivariateTE or ActiveInformationStorage.

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data_properties : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were z-standardised
                  before the estimation
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        self.settings = DotDict({})
        self.data_properties = DotDict({
            'n_nodes': n_nodes,
            'n_realisations': n_realisations,
            'normalised': normalised
        })

    def _print_edge_list(self, adjacency_matrix, weights):
        """Print edge list to console."""
        edge_list = adjacency_matrix.get_edge_list()
        if edge_list.size > 0:
            for e in edge_list:
                if weights == 'binary':
                    print('\t{0} -> {1}'.format(e[0], e[1]))
                else:
                    print('\t{0} -> {1}, {2}: {3}'.format(
                        e[0], e[1], weights, e[2]))
        else:
            print('No significant links found in the network.')

    def _check_result(self, process, settings):
        # Check if new result process is part of the network
        if process > (self.data_properties.n_nodes - 1):
            raise RuntimeError('Can not add single result - process {0} is not'
                               ' in no. nodes in the data ({1}).'.format(
                                   process, self.data_properties.n_nodes))
        # Don't add duplicate processes
        if self._is_duplicate_process(process):
            raise RuntimeError('Can not add single result - results for target'
                               ' or process {0} already exist.'.format(
                                   process))
        # Don't add results with conflicting settings
        if utils.conflicting_entries(self.settings, settings):
            raise RuntimeError(
                'Can not add single result - analysis settings are not equal.')

    def _is_duplicate_process(self, process):
        # Test if process is already present in object
        if process in self._processes_analysed:
            return True
        else:
            return False

    def combine_results(self, *results):
        """Combine multiple (partial) results objects.

        Combine a list of partial network analysis results into a single
        results object (e.g., results from analysis parallelized over
        processes). Raise an error if duplicate processes occur in partial
        results, or if analysis settings are not equal.

        Note that only conflicting settings cause an error (i.e., settings with
        equal keys but different values). If additional settings are included
        in partial results (i.e., settings with different keys) these settings
        are added to the common settings dictionary.

        Remove FDR-corrections from partial results before combining them. FDR-
        correction performed on the basis of parts of the network is not valid
        for the combined network.

        Args:
            results : list of Results objects
                single process analysis results from .analyse_network or
                .analyse_single_process methods, where each object contains
                partial results for one or multiple processes

        Returns:
            dict
                combined results object
        """
        for r in results:
            processes = r._processes_analysed
            if utils.conflicting_entries(self.settings, r.settings):
                raise RuntimeError('Can not combine results - analysis '
                                   'settings are not equal.')
            for p in processes:
                # Remove potential partial FDR-corrected results. These are no
                # longer valid for the combined network.
                if self._is_duplicate_process(p):
                    raise RuntimeError('Can not combine results - results for '
                                       'process {0} already exist.'.format(p))
                try:
                    del r.fdr_corrected
                    print('Removing FDR-corrected results.')
                except AttributeError:
                    pass

                try:
                    results_to_add = r._single_target[p]
                except AttributeError:
                    try:
                        results_to_add = r._single_process[p]
                    except AttributeError:
                        raise AttributeError(
                            'Did not find any method attributes to combine '
                            '(.single_proces or ._single_target).')
                self._add_single_result(p, results_to_add, r.settings)


class ResultsSingleProcessAnalysis(Results):
    """Store results of single process analysis.

    Provide a container for the results of algorithms for the analysis of
    individual processes (nodes) in a multivariate stochastic process,
    e.g., estimation of active information storage.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation:

    >>> res_network.settings.cmi_estimator

    or

    >>> res_network.settings['cmi_estimator'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data_properties : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were z-standardised
                  before estimation

        processes_analysed : list
            list of analysed processes
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self.processes_analysed = []
        self._single_process = {}
        self._single_process_fdr = DotDict()

    @property
    def processes_analysed(self):
        """Get index of the current_value."""
        return self._processes_analysed

    @processes_analysed.setter
    def processes_analysed(self, processes):
        self._processes_analysed = processes

    def _add_single_result(self, process, results, settings):
        """Add analysis result for a single process."""
        self._check_result(process, settings)
        self.settings.update(DotDict(settings))
        self._single_process[process] = DotDict(results)
        self.processes_analysed = list(self._single_process.keys())

    def _add_fdr(self, fdr, alpha=None, constant=None):
        """Add settings and results of FDR correction."""
        # Add settings of FDR-correction
        self.settings['alpha_fdr'] = alpha
        self.settings['fdr_constant'] = constant
        # Add results of FDR-correction. FDR-correction can be None if
        # correction is impossible due to the number of permutations in
        # individual analysis being too low to allow for individual p-values
        # to reach the FDR-thresholds. Add empty results in that case.
        if fdr is None:
            self._single_process_fdr = DotDict()
        else:
            self._single_process_fdr = DotDict(fdr)

    def get_single_process(self, process, fdr=True):
        """Return results for a single process in the network.

        Return results for individual processes, contains for each process

            - ais : float - AIS-value for current process
            - ais_pval : float - p-value of AIS estimate
            - ais_sign : bool - significance of AIS estimate wrt. to the
                alpha_mi specified in the settings
            - selected_var : list of tuples - variables with significant
                information about the current value of the process that have
                been added to the processes past state, a variable is
                described by the index of the process in the data and its lag
                in samples
            - current_value : tuple - current value used for analysis,
                described by target and sample index in the data

        Setting fdr to True returns FDR-corrected results (Benjamini, 1995).

        Args:
            process : int
                process id
            fdr : bool [optional]
                return FDR-corrected results, see documentation of network
                inference algorithms and stats.network_fdr (default=True)

        Returns:
            dict
                results for single process. Note that for convenience
                dictionary entries can either be accessed via keywords
                (result['selected_vars']) or via dot-notation
                (result.selected_vars).
        """
        # Return required key from required _single_process dictionary, dealing
        # with the FDR at a high level
        if process not in self.processes_analysed:
            raise RuntimeError('No results for process {0}.'.format(process))
        if fdr:
            try:
                return self._single_process_fdr[process]
            except AttributeError:
                raise RuntimeError(
                    'No FDR-corrected results have been added. Set'
                    ' ''fdr=False'' to see uncorrected results.')
            except KeyError:
                raise RuntimeError(
                    'No FDR-corrected results for process {0}. Set'
                    ' ''fdr=False'' to see uncorrected results.'.format(
                        process))
        else:
            try:
                return self._single_process[process]
            except AttributeError:
                raise RuntimeError('No results have been added.')
            except KeyError:
                raise RuntimeError(
                    'No results for process {0}.'.format(process))

    def get_significant_processes(self, fdr=True):
        """Return statistically-significant processes.

        Indicates for each process whether AIS is statistically significant
        (equivalent to the adjacency matrix returned for network inference)

        Args:
            fdr : bool [optional]
                return FDR-corrected results, see documentation of network
                inference algorithms and stats.network_fdr (default=True)

        Returns:
            numpy array
                Statistical significance for each process
        """
        significant_processes = np.array(
                [self.get_single_process(process=p, fdr=fdr)['ais_sign']
                 for p in self.processes_analysed],
                dtype=bool)
        return significant_processes


class ResultsNetworkAnalysis(Results):

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self._single_target = {}
        self.targets_analysed = []

    @property
    def targets_analysed(self):
        """Get index of the current_value."""
        return self._processes_analysed

    @targets_analysed.setter
    def targets_analysed(self, targets):
        self._processes_analysed = targets

    def _add_single_result(self, target, results, settings):
        """Add analysis result for a single target."""
        self._check_result(target, settings)
        # Add results
        self.settings.update(DotDict(settings))
        self._single_target[target] = DotDict(results)
        self.targets_analysed = list(self._single_target.keys())

    def get_single_target(self, target, fdr=True):
        """Return results for a single target in the network.

        Return results for individual processes, contains for each process

        Results for single targets include for each target

        - omnibus_te : float - TE-value for joint information transfer from all
          sources into the target
        - omnibus_pval : float - p-value of omnibus information transfer into
          the target
        - omnibus_sign : bool - significance of omnibus information transfer
          wrt. to the alpha_omnibus specified in the settings
        - selected_vars_sources : list of tuples - source variables with
          significant information about the current value
        - selected_vars_target : list of tuples - target variables with
          significant information about the current value
        - selected_sources_pval : array of floats - p-value for each selected
          variable
        - selected_sources_te : array of floats - TE-value for each selected
          variable
        - sources_tested : list of int - list of sources tested for the current
          target
        - current_value : tuple - current value used for analysis, described by
          target and sample index in the data

        Setting fdr to True returns FDR-corrected results (Benjamini, 1995).

        Args:
            target : int
                target id
            fdr : bool [optional]
                return FDR-corrected results, see documentation of network
                inference algorithms and stats.network_fdr (default=True)

        Returns:
            dict
                Results for single target. Note that for convenience
                dictionary entries can either be accessed via keywords
                (result['selected_vars_sources']) or via dot-notation
                (result.selected_vars_sources).
        """
        if target not in self.targets_analysed:
            raise RuntimeError('No results for target {0}.'.format(target))
        if fdr:
            try:
                return self._single_target_fdr[target]
            except AttributeError:
                raise RuntimeError(
                    'No FDR-corrected results have been added. Set'
                    ' ''fdr=False'' to see uncorrected results.')
            except KeyError:
                raise RuntimeError(
                    'No FDR-corrected results for target {0}. Set'
                    ' ''fdr=False'' to see uncorrected results.'.format(
                        target))
        else:
            try:
                return self._single_target[target]
            except AttributeError:
                raise RuntimeError('No results have been added.')
            except KeyError:
                raise RuntimeError(
                    'No results for target {0}.'.format(target))

    def get_target_sources(self, target, fdr=True):
        """Return list of sources (parents) for given target.

        Args:
            target : int
                target index
            fdr : bool [optional]
                if True, sources are returned for FDR-corrected results
                (default=True)
        """
        v = self.get_single_target(target, fdr)['selected_vars_sources']
        return np.unique(np.array([s[0] for s in v]))


class ResultsNetworkInference(ResultsNetworkAnalysis):
    """Store results of network inference.

    Provide a container for results of network inference algorithms, e.g.,
    MultivariateTE or Bivariate TE.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation:

    >>> res_network.settings.cmi_estimator

    or

    >>> res_network.settings['cmi_estimator'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data_properties : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were z-standardised
                  before estimation

        targets_analysed : list
            list of analysed targets
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self._single_target_fdr = DotDict()

    def _add_fdr(self, fdr, alpha=None, correct_by_target=None, constant=None):
        """Add settings and results of FDR correction."""
        # Add settings of FDR-correction
        self.settings['alpha_fdr'] = alpha
        self.settings['fdr_correct_by_target'] = correct_by_target
        self.settings['fdr_constant'] = constant
        # Add results of FDR-correction. FDR-correction can be None if
        # correction is impossible due to the number of permutations in
        # individual analysis being too low to allow for individual p-values
        # to reach the FDR-thresholds. Add empty results in that case.
        if fdr is None:
            self._single_target_fdr = DotDict()
        else:
            self._single_target_fdr = DotDict(fdr)

    def _get_inference_measure(self, target):
        if 'selected_sources_te' in self._single_target[target]:
            return self._single_target[target].selected_sources_te
        elif 'selected_sources_mi' in self._single_target[target]:
            return self._single_target[target].selected_sources_mi
        else:
            raise KeyError('No entry with network inference measure found for '
                           'current target')

    def get_target_delays(self, target, criterion='max_te', fdr=True):
        """Return list of information-transfer delays for a given target.

        Return a list of information-transfer delays for a given target.
        Information-transfer delays are determined by the lag of the variable
        in a source past that has the highest information transfer into the
        target process. There are two ways of identifying the variable with
        maximum information transfer:

            a) use the variable with the highest absolute TE value (highest
               information transfer),
            b) use the variable with the smallest p-value (highest statistical
               significance).

        Args:
            target : int
                target index
            criterion : str [optional]
                use maximum TE value ('max_te') or p-value ('max_p') to
                determine the source-target delay (default='max_te')
            fdr : bool [optional]
                return FDR-corrected results (default=True)

        Returns:
            numpy array
                information-transfer delays for each source
        """
        sources = self.get_target_sources(target=target, fdr=fdr)
        delays = np.zeros(sources.shape[0]).astype(int)

        # Get the source index for each past source variable of the target
        all_vars_sources = np.array([x[0] for x in self.get_single_target(
            target=target, fdr=fdr)['selected_vars_sources']])
        # Get the lag for each past source variable of the target
        all_vars_lags = np.array([x[1] for x in self.get_single_target(
            target=target, fdr=fdr)['selected_vars_sources']])
        # Get p-values and TE-values for past source variable
        pval = self.get_single_target(
            target=target, fdr=fdr)['selected_sources_pval']
        measure = self._get_inference_measure(target)

        # Find delay for each source
        for (ind, s) in enumerate(sources):
            if criterion == 'max_p':
                # Find the minimum p-value amongst the variables in source s
                delays_ind = np.argmin(pval[all_vars_sources == s])
            elif criterion == 'max_te':
                # Find the maximum TE-value amongst the variables in source s
                delays_ind = np.argmax(measure[all_vars_sources == s])

            delays[ind] = all_vars_lags[all_vars_sources == s][delays_ind]

        return delays

    def get_adjacency_matrix(self, weights, fdr=True):
        """Return adjacency matrix.

        Return adjacency matrix resulting from network inference. The adjacency
        matrix can either be generated from FDR-corrected results or
        uncorrected results. Multiple options for the weight are available.

        Args:
            weights : str
                can either be

                - 'max_te_lag': the weights represent the source -> target
                   lag corresponding to the maximum tranfer entropy value
                   (see documentation for method get_target_delays for details)
                - 'max_p_lag': the weights represent the source -> target
                   lag corresponding to the maximum p-value
                   (see documentation for method get_target_delays for details)
                - 'vars_count': the weights represent the number of
                   statistically-significant source -> target lags
                - 'binary': return unweighted adjacency matrix with binary
                   entries

                   - 1 = significant information transfer;
                   - 0 = no significant information transfer.

            fdr : bool [optional]
                return FDR-corrected results (default=True)

        Returns:
            AdjacencyMatrix instance
        """
        adjacency_matrix = AdjacencyMatrix(self.data_properties.n_nodes, int)

        if weights == 'max_te_lag':
            for t in self.targets_analysed:
                sources = self.get_target_sources(target=t, fdr=fdr)
                delays = self.get_target_delays(target=t,
                                                criterion='max_te',
                                                fdr=fdr)
                adjacency_matrix.add_edge_list(
                    sources, np.ones(len(sources), dtype=int) * t, delays)
        elif weights == 'max_p_lag':
            for t in self.targets_analysed:
                sources = self.get_target_sources(target=t, fdr=fdr)
                delays = self.get_target_delays(target=t,
                                                criterion='max_p',
                                                fdr=fdr)
                adjacency_matrix.add_edge_list(
                    sources, np.ones(len(sources), dtype=int) * t, delays)
        elif weights == 'vars_count':
            for t in self.targets_analysed:
                single_result = self.get_single_target(target=t, fdr=fdr)
                sources = np.zeros(len(single_result.selected_vars_sources))
                weights = np.zeros(len(single_result.selected_vars_sources))
                for i, s in enumerate(single_result.selected_vars_sources):
                    sources[i] = s[0]
                    weights[i] += 1
                adjacency_matrix.add_edge_list(
                    sources, np.ones(len(sources), dtype=int) * t, weights)
        elif weights == 'binary':
            for t in self.targets_analysed:
                single_result = self.get_single_target(target=t, fdr=fdr)
                sources = np.zeros(
                    len(single_result.selected_vars_sources), dtype=int)
                weights = np.zeros(
                    len(single_result.selected_vars_sources), dtype=int)
                for i, s in enumerate(single_result.selected_vars_sources):
                    sources[i] = s[0]
                    weights[i] = 1
                adjacency_matrix.add_edge_list(
                    sources, np.ones(len(sources), dtype=int) * t, weights)
        else:
            raise RuntimeError('Invalid weights value')
        return adjacency_matrix

    def print_edge_list(self, weights, fdr=True):
        """Print results of network inference to console.

        Print edge list resulting from network inference to console.
        Output may look like this:

            >>> 0 -> 1, max_te_lag = 2
            >>> 0 -> 2, max_te_lag = 3
            >>> 0 -> 3, max_te_lag = 2
            >>> 3 -> 4, max_te_lag = 1
            >>> 4 -> 3, max_te_lag = 1

        The edge list can either be generated from FDR-corrected results
        or uncorrected results. Multiple options for the weight
        are available (see documentation of method get_adjacency_matrix for
        details).

        Args:
            weights : str
                link weights (see documentation of method get_adjacency_matrix
                for details)
            fdr : bool [optional]
                return FDR-corrected results (default=True)
        """
        adjacency_matrix = self.get_adjacency_matrix(weights=weights, fdr=fdr)
        self._print_edge_list(adjacency_matrix, weights=weights)


class ResultsPartialInformationDecomposition(ResultsNetworkAnalysis):
    """Store results of Partial Information Decomposition (PID) analysis.

    Provide a container for results of Partial Information Decomposition (PID)
    algorithms.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation:

    >>> res_pid._single_target[2].source_1

    or

    >>> res_pid._single_target[2].['source_1'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data_properties : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were z-standardised
                  before the estimation

        targets_analysed : list
            list of analysed targets
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)

    def get_single_target(self, target):
        """Return results for a single target in the network.

        Results for single targets include for each target

        - source_1 : tuple - source variable 1
        - source_2 : tuple - source variable 2
        - selected_vars_sources : list of tuples - source variables used in PID
          estimation
        - s1_unq : float - unique information in source 1
        - s2_unq : float - unique information in source 2
        - syn_s1_s2 : float - synergistic information in sources 1 and 2
        - shd_s1_s2 : float - shared information in sources 1 and 2
        - current_value : tuple - current value used for analysis, described by
          target and sample index in the data
        - [estimator-specific settings]

        Args:
            target : int
                target id

        Returns:
            dict
                Results for single target. Note that for convenience
                dictionary entries can either be accessed via keywords
                (result['selected_vars_sources']) or via dot-notation
                (result.selected_vars_sources).
        """
        return super(ResultsPartialInformationDecomposition,
                     self).get_single_target(target, fdr=False)


class ResultsNetworkComparison(ResultsNetworkAnalysis):
    """Store results of network comparison.

    Provide a container for results of network comparison algorithms.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation: res_network.settings.cmi_estimator
    or res_network.settings['cmi_estimator'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing

        data_properties : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were z-standardised
                  before the estimation

        surrogate_distribution : dict
            for each target, surrogate distributions used for testing of each
            link into the target
        targets_analysed : list
            list of analysed targets
        ab : dict
            for each target, list of comparison results for all links into the
            target; True if link in condition A > link in condition B
        pval : dict
            for each target, list of p-values for all compared links
        cmi_diff_abs : dict
            for each target, list of absolute difference in interaction measure
            for all compared links
        data_properties : dict
            information regarding the data used for analysis
        settings : dict
            settings used for comparison
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)

    def _add_results(self, union_network, results, settings):
        # Check if results have already been added to this instance.
        if self.settings:
            raise RuntimeWarning('Overwriting existing results.')
        # Add results
        self.settings = DotDict(settings)
        self.targets_analysed = union_network['targets_analysed']
        for t in self.targets_analysed:
            self._single_target[t] = DotDict(union_network._single_target[t])
        # self.max_lag = union_network['max_lag']
        self.surrogate_distributions = results['cmi_surr']
        self.ab = results['a>b']
        self.cmi_diff_abs = results['cmi_diff_abs']
        self.pval = results['pval']

    def get_adjacency_matrix(self, weights='comparison'):
        """Return adjacency matrix.

        Return adjacency matrix resulting from network inference.
        Multiple options for the weights are available.

        Args:
            weights : str [optional]
                can either be

                - 'union': all links in the union network, i.e., all
                  links that were tested for a difference

                or return information for links with a significant difference

                - 'comparison': True for links with a significant difference in
                   inferred effective connectivity (default)
                - 'pvalue': absolute differences in inferred effective
                   connectivity for significant links
                - 'diff_abs': absolute difference

        Returns:
            AdjacencyMatrix instance
        """
        # Note: right now, the network comparison work on the uncorrected
        # networks only. This may have to change in the future, in which case
        # the value for 'fdr' when accessing single target results or adjacency
        # matrices has to be taken from the analysis settings.
        if weights == 'comparison':
            adjacency_matrix = AdjacencyMatrix(
                self.data_properties.n_nodes, int)
            for t in self.targets_analysed:
                sources = self.get_target_sources(t)
                for i, s in enumerate(sources):
                    adjacency_matrix.add_edge(s, t, int(self.ab[t][i]))
        elif weights == 'union':
            adjacency_matrix = AdjacencyMatrix(
                self.data_properties.n_nodes, int)
            for t in self.targets_analysed:
                sources = self.get_target_sources(t)
                adjacency_matrix.add_edge_list(
                    sources, np.ones(len(sources), dtype=int) * t,
                    np.ones(len(sources), dtype=int))
        elif weights == 'diff_abs':
            adjacency_matrix = AdjacencyMatrix(
                self.data_properties.n_nodes, float)
            for t in self.targets_analysed:
                sources = self.get_target_sources(t)
                for (i, s) in enumerate(sources):
                    print(self.cmi_diff_abs)
                    adjacency_matrix.add_edge(s, t, self.cmi_diff_abs[t][i])
        elif weights == 'pvalue':
            adjacency_matrix = AdjacencyMatrix(
                self.data_properties.n_nodes, float)
            for t in self.targets_analysed:
                sources = self.get_target_sources(t)
                for (i, s) in enumerate(sources):
                    adjacency_matrix.add_edge(s, t, self.pval[t][i])
        else:
            raise RuntimeError('Invalid weights value')

        # self._print_edge_list(adjacency_matrix, weights=weights)
        return adjacency_matrix

    def print_edge_list(self, weights='comparison'):
        """Print results of network comparison to console.

        Print results of network comparison to console. Output looks like this:

            >>> 0 -> 1, diff_abs = 0.2
            >>> 0 -> 2, diff_abs = 0.5
            >>> 0 -> 3, diff_abs = 0.7
            >>> 3 -> 4, diff_abs = 1.3
            >>> 4 -> 3, diff_abs = 0.4

        indicating differences in the network inference measure for a link
        source -> target.

        Args:
            weights : str [optional]
                weights for the adjacency matrix (see documentation of method
                get_adjacency_matrix for details)
        """
        adjacency_matrix = self.get_adjacency_matrix(weights=weights)
        self._print_edge_list(adjacency_matrix, weights=weights)

    def get_single_target(self, target):
        """Return results for a single target in the network.

        Results for single targets include for each target

        - sources : list of ints - list of sources inferred for the current
          target (union of sources from both data sets entering the comparison)
        - selected_vars_sources : list of tuples - source variables with
          significant information about the current value (union of both
          conditions)
        - selected_vars_target : list of tuples - target variables with
          significant information about the current value (union of both
          conditions)

        Args:
            target : int
                target id

        Returns:
            dict
                Results for single target. Note that for convenience
                dictionary entries can either be accessed via keywords
                (result['selected_vars_sources']) or via dot-notation
                (result.selected_vars_sources).
        """
        return super(ResultsNetworkComparison, self).get_single_target(
            target, fdr=False)

    def get_target_sources(self, target):
        """Return list of sources (parents) for given target.

        Args:
            target : int
                target index
        """
        v = self.get_single_target(target)['selected_vars_sources']
        return np.unique(np.array([s[0] for s in v]))
