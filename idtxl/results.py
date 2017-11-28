"""Provide results class for IDTxl network analysis.

Created on Wed Sep 20 18:37:27 2017

@author: patricia
"""
import copy as cp
import itertools as it
import numpy as np
from . import idtxl_utils as utils
from . import idtxl_exceptions as ex
try:
    import networkx as nx
except ImportError as err:
    ex.package_missing(
        err,
        ('networkx is not available on this system. Install it from '
         'https://pypi.python.org/pypi/networkx/2.0 to export and plot IDTxl '
         'results in this format.'))


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


class Results():
    """Parent class for results of network analysis algorithms.

    Provide a container for results of network analysis algorithms, e.g.,
    MultivariateTE or ActiveInformationStorage.

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were normalised before
                  estimation
                - single
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        self.settings = DotDict({})
        self.data = DotDict({
            'n_nodes': n_nodes,
            'n_realisations': n_realisations,
            'normalised': normalised
        })

    def _print_to_console(self, adjacency_matrix, measure):
        """Print adjacency matrix to console."""
        link_found = False
        for s in range(self.data.n_nodes):
            for t in range(self.data.n_nodes):
                if adjacency_matrix[s, t]:
                    print('\t{0} -> {1}, {2}: {3}'.format(
                        s, t, measure, adjacency_matrix[s, t]))
                    link_found = True
        if not link_found:
            print('No significant links in network.')

    def _export_to_networkx(self, adjacency_matrix):
        """Create networkx DiGraph object from numpy adjacency matrix."""
        return nx.from_numpy_matrix(
            adjacency_matrix, create_using=nx.DiGraph())

    def _check_result(self, process, settings):
        # Check if new result process is part of the network
        if process > (self.data.n_nodes - 1):
            raise RuntimeError('Can not add single result - process {0} is not'
                               ' in no. nodes in the data ({1}).'.format(
                                   process, self.data.n_nodes))
        # Don't add duplicate processes
        if self._is_duplicate_process(process):
            raise RuntimeError('Can not add single result - results for target'
                               ' or process {0} already exist.'.format(
                                   process))
        # Dont' add results with conflicting settings
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
        correction performed on the basis of parts of the network a not valid
        for the combined network.

        Args:
            results : list of Results objects
                single process analysis results from .analyse_network or
                .analyse_single_process methods, where each object contains
                partial results for one or multiple processes

        Returns:
            dict
                combined results dict
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
                    results_to_add = r.single_target[p]
                except AttributeError:
                    try:
                        results_to_add = r.single_process[p]
                    except AttributeError:
                        raise AttributeError(
                            'Did not find any method attributes to combine '
                            '(.single_proces or .single_target).')
                self._add_single_result(p, results_to_add, r.settings)


class ResultsSingleProcessAnalysis(Results):
    """Store results of single process analysis.

    Provide a container for results of algorithms for the analysis of single
    processes forming network nodes, e.g., estimation of active information
    storage.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation: res_network.single_target[2].omnibus_pval
    or res_network.single_target[2].['omnibus_pval'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were normalised before
                  estimation
                - single

        single_process : dict
            results for individual processes, contains for each process

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

        processes_analysed : list
            list of processes analyzed
        significant_processes : np array
            indicates for each process whether AIS is significant
        fdr_correction : dict
            FDR-corrected results, see documentation of network inference
            algorithms and stats.network_fdr

    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self.single_process = {}
        self.processes_analysed = []
        self.significant_processes = np.zeros(self.data.n_nodes, dtype=bool)
        self._add_fdr(None)

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
        self.single_process[process] = DotDict(results)
        self.processes_analysed = list(self.single_process.keys())
        self._update_significant_processes(process)

    def _add_fdr(self, fdr):
        # Add results of FDR-correction
        if fdr is None:
            self.fdr_correction = DotDict()
        else:
            self.fdr_correction = DotDict(fdr)
            self.fdr_correction.significant_processes = np.zeros(
                self.data.n_nodes, dtype=bool)
            self._update_significant_processes(fdr=True)

    def _update_significant_processes(self, process=None, fdr=False):
        """Update list of processes with significant results."""
        # If no process is given, build list from scratch, else: just update
        # the requested process to save time.
        if process is None:
            update_processes = self.processes_analysed
        else:
            update_processes = [process]

        for p in update_processes:
            if fdr:
                self.fdr_correction.significant_processes[p] = (
                    self.fdr_correction.single_process[p].ais_sign)
            else:
                self.significant_processes[p] = self.single_process[p].ais_sign


class ResultsNetworkAnalysis(Results):

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self.single_target = {}
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
        self.single_target[target] = DotDict(results)
        self.targets_analysed = list(self.single_target.keys())
        self._update_adjacency_matrix(target=target)

    def _update_adjacency_matrix(self, target=None, fdr=False):
        """Update adjacency matrix."""
        # If no target is given, build adjacency matrix from scratch, else:
        # just update the requested target to save time.
        if target is None:
            update_targets = self.targets_analysed
        else:
            update_targets = [target]

        for t in update_targets:
            sources = self.get_target_sources(target=t, fdr=fdr)
            delays = self.get_target_delays(target=t, fdr=fdr)
            if sources.size:
                if fdr:
                    self.fdr_correction.adjacency_matrix[sources, t] = delays
                else:
                    self.adjacency_matrix[sources, t] = delays

    def get_target_sources(self, target, fdr=False):
        """Return list of sources for given target.

        Args:
            target : int
                target index
            fdr : bool [optional]
                if True, sources are returned for FDR-corrected results
        """
        if target not in self.targets_analysed:
            raise RuntimeError('No results for target {0}.'.format(target))
        if fdr:
            try:
                return np.unique(np.array(
                    [s[0] for s in (self.fdr_correction[target].
                                    selected_vars_sources)]))
            except AttributeError:
                raise RuntimeError('No FDR-corrected results have been added.')
            except KeyError:
                RuntimeError(
                    'Didn''t find results for target {0}.'.format(target))
        else:
            try:
                return np.unique(np.array(
                    [s[0] for s in (self.single_target[target].
                                    selected_vars_sources)]))
            except AttributeError:
                raise RuntimeError('No results have been added.')
            except KeyError:
                raise RuntimeError(
                    'Didn''t find results for target {0}.'.format(target))


class ResultsNetworkInference(ResultsNetworkAnalysis):
    """Store results of network inference.

    Provide a container for results of network inference algorithms, e.g.,
    MultivariateTE or Bivariate TE.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation: res_network.single_target[2].omnibus_pval
    or res_network.single_target[2].['omnibus_pval'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were normalised before
                  estimation
                - single

        adjacency_matrix : 2D numpy array
            adjacency matrix describing the inferred network structure, if
            applicable entries denote the information-transfer delay
        single_target : dict
            results for individual targets, contains for each target

                - omnibus_te : float - TE-value for joint information transfer
                  from all sources into the target
                - omnibus_pval : float - p-value of omnibus information
                  transfer into the target
                - omnibus_sign : bool - significance of omnibus information
                  transfer wrt. to the alpha_omnibus specified in the settings
                - selected_vars_sources : list of tuples - source variables
                  with significant information about the current value
                - selected_vars_target : list of tuples - target variables
                  with significant information about the current value
                - selected_sources_pval : array of floats - p-value for each
                  selected variable
                - selected_sources_te : array of floats - TE-value for each
                  selected variable
                - sources_tested : list of int - list of sources tested for the
                  current target
                - current_value : tuple - current value used for analysis,
                  described by target and sample index in the data

        targets_analysed : list
            list of targets analyzed
        fdr_correction : dict
            FDR-corrected results, see documentation of network inference
            algorithms and stats.network_fdr

    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self.adjacency_matrix = np.zeros(
            (self.data.n_nodes, self.data.n_nodes), dtype=int)
        self._add_fdr(None)

    def get_target_delays(self, target, find_delay='max_te', fdr=False):
        """Return list of information-transfer delays for given target.

        Return a list of information-transfer delays for given target.
        Information-transfer delays are determined by the lag of the variable
        in a sources' past that has the highest information transfer into the
        target process. There are two ways of idendifying the variable with
        maximum information transfer:

            a) use the variable with the highest absolute TE value (highest
               information transfer),
            b) use the variable with the smallest p-value (highest statistical
               significance).

        Args:
            target : int
                target index
            fdr : bool [optional]
                print FDR-corrected results (default=False)
            find_delay : str [optional]
                use maximum TE value ('max_te') or p-value ('max_p') to
                determine the source-target delay (default='max_te')

        Returns:
            numpy array
                Information-transfer delays for each source
        """
        sources = self.get_target_sources(target, fdr)
        delays = np.zeros(sources.shape[0]).astype(int)

        # Get the source index for each past source variable of the target
        all_vars_sources = np.array(
            [x[0] for x in self.single_target[target].selected_vars_sources])
        # Get the lag for each past source variable of the target
        all_vars_lags = np.array(
            [x[1] for x in self.single_target[target].selected_vars_sources])
        # Get p-values and TE-values for past source variable
        pval = self.single_target[target].selected_sources_pval
        te = self.single_target[target].selected_sources_te

        # Find delay for each source
        for (ind, s) in enumerate(sources):
            if find_delay == 'max_p':
                # Find the minimum p-value amongst the variables in source s
                delays_ind = np.argmin(pval[all_vars_sources == s])
            elif find_delay == 'max_te':
                # Find the maximum TE-value amongst the variables in source s
                delays_ind = np.argmax(te[all_vars_sources == s])

            delays[ind] = all_vars_lags[all_vars_sources == s][delays_ind]

        return delays

    def export_networkx_graph(self, fdr=False):
        """Generate networkx graph object for an inferred network.

        Generate a weighted, directed graph object from the network of inferred
        (multivariate) interactions (e.g., multivariate TE), using the networkx
        class for directed graphs (DiGraph). The graph is weighted by the
        reconstructed source-target delays (see documentation for method
        get_target_delays for details).

        Args:
            fdr : bool [optional]
                return FDR-corrected results

        Returns:
            DiGraph object
                instance of a directed graph class from the networkx
                package (DiGraph)
        """
        if fdr:
            return self._export_to_networkx(
                self.fdr_correction.adjacency_matrix)
        else:
            return self._export_to_networkx(self.adjacency_matrix)

    def export_networkx_source_graph(self, target,
                                     sign_sources=True, fdr=False):
        """Generate graph object of source variables for a single target.

        Generate a graph object from the network of (multivariate)
        interactions (e.g., multivariate TE) between single source variables
        and a target process using the networkx class for directed graphs
        (DiGraph). The graph shows the information transfer between individual
        source variables and the target.

        Args:
            target : int
                target index
            sign_sources : bool [optional]
                add only sources significant information contribution
                (default=True)
            fdr : bool [optional]
                return FDR-corrected results

        Returns:
            DiGraph object
                instance of a directed graph class from the networkx
                package (DiGraph)
        """
        if target not in self.targets_analysed:
            raise RuntimeError('No results for target {0}.'.format(target))
        graph = nx.DiGraph()
        # Add the target as a node. Each node is a tuple with format (process
        # index, sample index).
        graph.add_node(self.single_target[target].current_value)

        # Get list of *all* past variables.
        if fdr:
            if (self.fdr_correction.adjacency_matrix == 0).all():
                all_variables = []  # no sign. links after FDR-correction
            else:
                all_variables = (
                    self.fdr_correction.single_target[target].selected_vars_sources +
                    self.fdr_correction.single_target[target].selected_vars_target)
        else:
            all_variables = (
                self.single_target[target].selected_vars_sources +
                self.single_target[target].selected_vars_target)

        if sign_sources:  # Add only significant past variables as nodes.
            graph.add_nodes_from(all_variables)
        else:   # Add all tested past variables as nodes.
            # Get all sample indices.
            current_value = self.single_target[target].current_value
            min_lag = self.settings.min_lag_sources
            max_lag = self.settings.max_lag_sources
            tau = self.settings.max_lag_sources
            samples_tested = np.arange(
                current_value[1] - min_lag, current_value[1] - max_lag, -tau)
            # Get source indices
            sources_tested = self.single_target[target].sources_tested
            # Create tuples from source and sample indices
            nodes = [i for i in it.product(sources_tested, samples_tested)]
            graph.add_nodes_from(nodes)

        # Add edges from significant past variables to the target. Here, one
        # could add additional info in the future, networkx graphs support
        # attributes for graphs, nodes, and edges.
        for v in all_variables:
            graph.add_edge(v, self.single_target[target].current_value)

        return graph

    def print_to_console(self, fdr=False):
        """Print results of network inference to console.

        Print results of network inference to console. Output looks like this:

            0 -> 1, u = 2
            0 -> 2, u = 3
            0 -> 3, u = 2
            3 -> 4, u = 1
            4 -> 3, u = 1

        indicating significant information transfer source -> target with an
        information-transfer delay u (see documentation for method
        get_target_delays for details). The network can either be plotted from
        FDR-corrected results or uncorrected results.

        Args:
            fdr : bool [optional]
                print FDR-corrected results (default=False)
        """
        if fdr:
            adjacency_matrix = self.fdr_correction.adjacency_matrix
        else:
            adjacency_matrix = self.adjacency_matrix
        self._print_to_console(adjacency_matrix, 'u')

    def _add_fdr(self, fdr):
        # Add results of FDR-correction
        if fdr is None:
            self.fdr_correction = DotDict()
            self.fdr_correction['adjacency_matrix'] = np.zeros(
                (self.data.n_nodes, self.data.n_nodes), dtype=int)
        else:
            self.fdr_correction = DotDict(fdr)
            self.fdr_correction.adjacency_matrix = np.zeros(
                (self.data.n_nodes, self.data.n_nodes), dtype=int)
            self._update_adjacency_matrix(fdr=True)


class ResultsPartialInformationDecomposition(ResultsNetworkAnalysis):
    """Store results of network inference.

    Provide a container for results of network inference algorithms, e.g.,
    MultivariateTE or Bivariate TE.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation: res_network.single_target[2].omnibus_pval
    or res_network.single_target[2].['omnibus_pval'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        data : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were normalised before
                  estimation
                - single

        single_target : dict
            results for individual targets, contains for each target

                - source_1 : tuple - source variable 1
                - source_2 : tuple - source variable 2
                - s1_unq : float - unique information in source 1
                - s2_unq : float - unique information in source 2
                - syn_s1_s2 : float - synergistic information in sources 1
                  and 2
                - shd_s1_s2 : float - shared information in sources 1 and 2
                - s1_unq_sign : float - TODO
                - s2_unq_sign : float - TODO
                - s1_unq_p_val : float - TODO
                - s2_unq_p_val : float - TODO
                - syn_sign : float - TODO
                - syn_p_val : float - TODO
                - shd_sign : float - TODO
                - shd_p_val : float - TODO
                - current_value : tuple - current value used for analysis,
                  described by target and sample index in the data


        targets_analysed : list
            list of targets analyzed
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self.adjacency_matrix = np.zeros(
            (self.data.n_nodes, self.data.n_nodes), dtype=int)

    def _update_adjacency_matrix(self, target):
        sources = self.get_target_sources(target)
        if sources.size:
            self.adjacency_matrix[sources, target] = 1


class ResultsNetworkComparison(ResultsNetworkAnalysis):
    """Store results of network comparison.

    Provide a container for results of network comparison algorithms.

    Note that for convenience all dictionaries in this class can additionally
    be accessed using dot-notation: res_network.single_target[2].omnibus_pval
    or res_network.single_target[2].['omnibus_pval'].

    Attributes:
        settings : dict
            settings used for estimation of information theoretic measures and
            statistical testing
        union_network : dict
            union of networks that entered the comparison
        adjacency_matrix : 2D numpy array
            adjacency matrix describing the differences in inferred effective
            connectivity
        data : dict
            data properties, contains

                - n_nodes : int - total number of nodes in the network
                - n_realisations : int - number of samples available for
                  analysis given the settings (e.g., a high maximum lag used in
                  network inference, results in fewer data points available for
                  estimation)
                - normalised : bool - indicates if data were normalised before
                  estimation
                - single

        comparison : dict
            results of network comparison, contains

                - cmi_diff_abs :
                - a>b :
                - cmi_surr :
                - sign :
                - p_val :

        targets_analysed : list
            list of targets analyzed
    """

    def __init__(self, n_nodes, n_realisations, normalised):
        super().__init__(n_nodes, n_realisations, normalised)
        self.adjacency_matrix_pvalue = np.ones(
            (self.data.n_nodes, self.data.n_nodes), dtype=float)
        self.adjacency_matrix_comparison = np.zeros(
            (self.data.n_nodes, self.data.n_nodes), dtype=bool)
        self.adjacency_matrix_union = np.zeros(
            (self.data.n_nodes, self.data.n_nodes), dtype=int)
        self.adjacency_matrix_diff_abs = np.zeros(
            (self.data.n_nodes, self.data.n_nodes), dtype=float)

    def _add_results(self, union_network, results, settings):
        # Check if results have already been added to this instance.
        if self.settings:
            raise RuntimeWarning('Overwriting existing results.')
        # Add results
        self.settings = DotDict(settings)
        self.targets_analysed = union_network['targets_analysed']
        for t in self.targets_analysed:
            self.single_target[t] = DotDict(union_network.single_target[t])
        # self.max_lag = union_network['max_lag']
        self.surrogate_distributions = results['cmi_surr']
        self._update_adjacency_matrix(results)

    def _update_adjacency_matrix(self, results):
        for t in self.targets_analysed:
            sources = self.get_target_sources(t)
            if sources.size:
                self.adjacency_matrix_union[sources, t] = 1
            for (i, s) in enumerate(sources):
                self.adjacency_matrix_comparison[s, t] = results['a>b'][t][i]
                self.adjacency_matrix_diff_abs[s, t] = (
                    results['cmi_diff_abs'][t][i])
                self.adjacency_matrix_pvalue[s, t] = results['pval'][t][i]

    def print_to_console(self, matrix='comparison'):
        """Print results of network comparison to console.

        Print results of network comparison to console. Output looks like this:

            0 -> 1, abs_diff = 0.2
            0 -> 2, abs_diff = 0.5
            0 -> 3, abs_diff = 0.7
            3 -> 4, abs_diff = 1.3
            4 -> 3, abs_diff = 0.4

        indicating differences in the network inference measure for a link
        source -> target.

        Args:
            matrix : str [optional]
                can either be

                - 'union': print all links in the union network, i.e., all
                  links that were tested for a difference

                or print information for links with a significant difference

                - 'comparison': links with a significant difference (default)
                - 'pvalue': print p-values for links with a significant
                   difference
                - 'diff_abs': print the absolute difference
        """
        if matrix == 'comparison':
            adjacency_matrix = self.adjacency_matrix_comparison
        elif matrix == 'union':
            adjacency_matrix = self.adjacency_matrix_union
        elif matrix == 'diff_abs':
            adjacency_matrix = self.adjacency_matrix_diff_abs
        elif matrix == 'pvalue':
            adjacency_matrix = self.adjacency_matrix_pvalue
        self._print_to_console(adjacency_matrix, matrix)

    def export_networkx_graph(self, matrix):
        """Generate networkx graph object from network comparison results.

        Generate a weighted, directed graph object from the adjacency matrix
        representing results of network comparison, using the networkx class
        for directed graphs (DiGraph). The graph is weighted by the
        results requested by 'matrix'.

        Args:
            matrix : str [optional]
                can either be
                - 'union': print all links in the union network, i.e., all
                  links that were tested for a difference
                or print information for links with a significant difference
                - 'comparison': links with a significant difference (default)
                - 'pvalue': print p-values for links with a significant
                   difference
                - 'diff_abs': print the absolute difference

        Returns:
            DiGraph object
                instance of a directed graph class from the networkx
                package (DiGraph)
        """
        if matrix == 'comparison':
            adjacency_matrix = self.adjacency_matrix_comparison
        elif matrix == 'union':
            adjacency_matrix = self.adjacency_matrix_union
        elif matrix == 'diff_abs':
            adjacency_matrix = self.adjacency_matrix_diff_abs
        elif matrix == 'pvalue':
            adjacency_matrix = self.adjacency_matrix_pvalue
        return self._export_to_networkx(adjacency_matrix)
