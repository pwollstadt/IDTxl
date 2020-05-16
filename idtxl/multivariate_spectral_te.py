"""Perform network inference using spectral multivarate transfer entropy.

Note:
    Written for Python 3.4+
"""
from .network_inference_spectral import NetworkInferenceSpectral
from .results import ResultsSpectralTE


class MultivariateSpectralTE(NetworkInferenceSpectral):
    """Perform network inference using multivariate spectral transfer entropy.

    Set parameters necessary for inference of spectral components of
    multivariate transfer entropy (TE). To perform network inference call
    analyse_network() or analyse_single_target() on an instance of the data
    class and the results from a multivarate TE analysis.

    References:
    - Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for
      nonlinearity tests. Phys Rev Lett, 77(4):635.
      https://doi.org/10.1103/PhysRevLett.77.635
    - Keylock, C.J. (2010). Characterizing the structure of nonlinear systems
      using gradual wavelet reconstruction. Nonlinear Proc Geoph.
      17(6):615–632. https://doi.org/10.5194/npg-17-615-2010
    - Percival, D.B. & Walden, A.T. (2000). Wavelet Methods for Time Series
      Analysis. Cambridge: Cambridge University Press.
    """

    def __init__(self):
        super().__init__()

    def analyse_network(self, settings, data, results, targets='all',
                        sources='all'):
        """Find multivariate spectral transfer entropy between all nodes.

        Estimate spectral transfer entropy (TE) for all or selected significant
        links in the inferred network.

        Note:
            For more details on the estimation of multivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

        Args:
            settings : dict
                parameters for estimation and statistical testing, see
                documentation of analyse_single_target() for details, settings
                can further contain

                - verbose : bool [optional] - toggle console output
                  (default=True)

            data : Data instance
                raw data from which the network was inferred
            results : ResultsNetworkInference() instance
                results from multivariate network inference, e.g., using TE
            targets : list of int | 'all' [optional]
                index of target processes, either all targets with significant
                 information transfer or a subset of significant targets
                 (default='all')
            sources : list of int | list of list | 'all' [optional]
                indices of significant source processes for each target
                (default='all');
                if 'all', for each link, all significant sources are tested
                if list of int, significant sources specified in the list are
                tested;
                if list of lists, sources specified in each inner list are
                tested for the corresponding target

        Returns:
            ResultsSpectralTE() instance
                results of spectral analysis, see documentation of
                ResultsSpectralTE()
        """
        settings.setdefault('verbose', True)
        print(targets)
        if targets == 'all':
            targets = results.targets_analysed
        if sources == 'all':
            sources = ['all' for t in targets]
        if (type(sources) is list) and (type(sources[0]) is int):
            sources = [sources for t in targets]
        if (type(sources) is list) and (type(sources[0]) is list):
            pass
        else:
            ValueError('Sources was not specified correctly: {0}.'.format(
                                                                    sources))
        assert(len(sources) == len(targets)), ('List of targets and list of '
                                               'sources have to have the same '
                                               'same length')

        # Perform spectral TE estimation for each target individually
        results_spec = ResultsSpectralTE(n_nodes=data.n_processes,
                                         n_realisations=data.n_realisations(),
                                         normalised=data.normalise)
        for t in range(len(targets)):
            if settings['verbose']:
                print('\n####### analysing target with index {0} from list {1}'
                      .format(t, targets))
            res_single = self.analyse_single_target(
                    settings, data, results, targets[t], sources[t])
            results_spec.combine_results(res_single)

        # Get no. realisations actually used for estimation from single target
        # analysis.
        results_spec.data_properties.n_realisations = (
            res_single.data_properties.n_realisations)

        return results_spec

    def analyse_single_target(self, settings, data, results, target,
                              sources='all'):
        """Find multivariate spectral transfer entropy into a target.

        Test spectral transfer entropy (TE) for all inferred links for a single
        target, in a frequency band defined by scale:

        (1) pick next identified source s in res_target
        (2) perform a maximal overlap discrete wavelet transform (MODWT)
        (3) destroy information carried by a single frequency band by
            scrambling the coefficients in the respective scale
        (4) perform the inverse of the MODWT, iMODWT, to get back the time-
            domain representation of the variable
        (5) calculate multivariate TE between s and the target, conditional on
            all other relevant sources
        (6) repeat (3) to (5) n_perm number of times to build a test
            distribution
        (7) test original multivariate TE against the test distribution

        Modwt coefficients at scale j are associated to the same nominal
        frequency band |f| = [1/2 .^ j + 1, 1/2 .^ j] (see A. Walden "Wavelet
        Analysis of Discrete Time Series"). For example, for a 1000 Hz signal

        * scale 4 is equivalent to
        ([1/2 .^ 5, 1/2.^4]) * 1000 = [31.25 Hz 62.5 Hz]  # gamma band
        * scale 5 is equivalent to
        ([1/2 .^ 6, 1/2.^5]) * 1000 = [15.62 Hz 31.25 Hz]  # beta band
        * scale 6 is equivalent to
        ([1/2 .^ 7, 1/2.^6]) * 1000 = [7.81 Hz 15.62 Hz]   # alpha band
        * scale 7 is equivalent to
        ([1/2 .^ 7, 1/2.^6]) * 1000 = [3.9 Hz 7.81 Hz]   # theta band

        Note:
            For more details on the estimation of multivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

        Args:
            settings : dict
                additional parameters for estimation and statistical testing of
                spectral TE, settings from previous multivariate TE estimation
                are reused, settings can contain

                - spectral_analysis_type : str [optional] - destroy only the
                  'source', only the 'target', or 'both' for surrogate
                  creation, or 'SOSO' for 'swap-out swap-out' algorithm
                  (default='both')
                - wavelet : str [optional] - mother wavelet used for spectral
                  decomposition (default='db16')
                - n_scale : int [optional] - number of scales for decomposition
                  (default='max')
                - fdr_corrected : bool [optional] - use FDR-corrected results
                  (default=True)
                - n_perm_spec : int [optional] - number of permutations
                  (default=200)
                - alpha_spec : float [optional] - critical alpha level for
                  statistical significance (default=0.05)
                - permute_in_time : bool [optional] - force surrogate
                  creation by shuffling realisations in time instead of
                  shuffling replications; see documentation of
                  Data.permute_samples() for further settings (default=False)
                - verbose : bool [optional] - toggle console output
                  (default=True)
                - parallel_surr:bool -create surrogates in parallel
                  (default=False)
                - surr_type: string - 'spectr' or 'iaaft', for details see
                  spectral_surrogates() and GWS_surrogates().

            data : Data instance
                raw data from which the network was inferred
            results : ResultsNetworkInference() instance
                results from multivariate network inference, e.g., using TE
            target : int
                index of target process
            sources : list of int | 'all' [optional]
                indices of significant source processes for target
                (default='all');
                if 'all', for each link, all significant sources are tested
                if list of int, significant sources specified in the list are
                tested;

        Returns:
            ResultsSpectralTE() instance
                results of spectral analysis, see documentation of
                ResultsSpectralTE()
        """
        # Check input.
        self._initialise(settings, results, data, sources, target)

        # Main algorithm.
        results_spec = {}
        if self.settings['spectral_analysis_type'] == 'source':
            if self.settings['verbose']:
                print('Spectral analysis with source surrogates')
            for current_scale in range(0, self.settings['n_scale']):
                print('Testing Scale n: {0}'.format(current_scale))
                res_scale = self._spectral_analysis_source(
                    data, results, current_scale)
                results_spec[current_scale] = res_scale

        elif self.settings['spectral_analysis_type'] == 'target':
            print('Spectral analysis with target surrogates')
            for current_scale in range(0, self.settings['n_scale']):
                # combine result for each scale  e.g. combine result
                print('Testing Scale n: {0}' .format(current_scale))
                res_scale = self._spectral_analysis_target(
                    data, results, current_scale)
                results_spec[current_scale] = res_scale

        elif self.settings['spectral_analysis_type'] == 'both':
            print('Spectral analysis with source surrogates')
            results_spec['target'] = {}
            results_spec['source'] = {}
            for current_scale in range(0, self.settings['n_scale']):
                print('Testing Scale n: {0}'.format(current_scale))
                res_scale = self._spectral_analysis_source(
                    data, results, current_scale)
                results_spec['source'][current_scale] = res_scale

            print('Spectral analysis with target surrogates')
            for current_scale in range(0, self.settings['n_scale']):
                print('Testing Scale n: {0}'.format(current_scale))
                res_scale = self._spectral_analysis_target(
                    data, results, current_scale)
                results_spec['target'][current_scale] = res_scale

        elif self.settings['spectral_analysis_type'] == 'SOSO':
            print('Spectral analysis with SOS surrogates')
            results_spec['target'] = {}
            results_spec['source'] = {}
            scale_target = self.settings['scale_target']
            for current_scale in self.settings['scale_source']:
                print('Testing Scale n: {0}'.format(current_scale))
                res_scale = self._spectral_analysis_SOS_T(
                    data, results, current_scale, scale_target)
                results_spec['source'][current_scale] = res_scale

        else:
            raise ValueError('Unkown spectral_analysis_type {0}.'.format(
                self.settings['spectral_analysis_type']))

        # Add analyis info.
        results = ResultsSpectralTE(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(self.current_value),
            normalised=data.normalise)
        results._add_single_result(
            settings=self.settings,
            target=self._target,
            results=results_spec)

        return results
