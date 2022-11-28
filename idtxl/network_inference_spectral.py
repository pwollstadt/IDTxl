import copy as cp
import numpy as np
from joblib import Parallel, delayed
from . import stats
from .network_analysis import NetworkAnalysis
from .estimator import find_estimator
from .results import DotDict
from .stats import _generate_spectral_surrogates
from . import modwt
from .data import Data
from . import idtxl_exceptions as ex
from . import idtxl_utils as utils


class NetworkInferenceSpectral(NetworkAnalysis):
    """Parent class for spectral network inference algorithms.

    Hold variables that are relevant for spectral network inference using, for
    example, multivariate transfer entropy.

    Attributes:
        settings : dict
            settings for estimation of information theoretic measures and
            statistical testing, see child classes for documentation
        target : int
            target process of analysis
        current_value : tuple
            index of the current value
        selected_vars_full : list of tuples
            indices of the full set of random variables to be conditioned on
        selected_vars_target : list of tuples
            indices of the set of conditionals coming from the target process
        selected_vars_sources : list of tuples
            indices of the set of conditionals coming from source processes
    """

    def __init__(self):
        # Create class attributes for estimation
        self.te_surrogate = None
        self.deltaO = None
        self.temporal_surrogate = None
        self.pvalues = None
        self.significance = None
        super().__init__()

    def _initialise(self, settings, results, data, sources, target):
        # Set estimator in the child class for network inference because the
        # estimated quantity may be different from CMI in other inference
        # algorithms. (Everything else can be done in the parent class.)
        try:
            EstimatorClass = find_estimator(results.settings['cmi_estimator'])
        except KeyError:
            raise KeyError('Estimator was not specified!')
        # Don't add results with conflicting settings
        if utils.conflicting_entries(results.settings, settings):
            raise RuntimeError(
                'Conflicting entries in spectral TE and network inference '
                'settings.')
        self.settings = settings.copy()
        self.settings.update(DotDict(results.settings))  # combine inference and spec TE settings
        self.settings.setdefault('verbose', True)
        self.settings.setdefault('spectral_analysis_type', 'both')
        self.settings.setdefault('wavelet', 'la16')
        self.settings.setdefault('fdr_corrected', True)
        self.settings.setdefault('perm_in_time_spec', True)
        self.settings.setdefault('perm_type_spec', 'random')
        self.max_scale = int(np.log2(data.n_samples))
        self.settings.setdefault('n_scale', self.max_scale)
        self.settings.setdefault('n_perm_spec', 200)
        self.settings.setdefault('alpha_spec', 0.05)
        self.settings.setdefault('parallel_surr', False)
        self.settings['tail_spec'] = 'one'
        self._cmi_estimator = EstimatorClass(settings)

        # Check scale
        if type(self.settings['n_scale']) is not int:
            raise TypeError('n_scale must be an integer')

        assert (self.settings['n_scale'] <= self.max_scale), (
            'scale ({0}) must be smaller or equal to max_scale'
            ' ({1}).'.format(self.settings['n_scale'], self.max_scale))
        if self.settings['verbose']:
            print('Max. scale is {0}, requested scale is {1}.'.format(
                                    self.max_scale, self.settings['n_scale']))

        if self.settings['perm_type_spec'] == 'block':
            self.settings.setdefault(
                'block_size', int(np.round(data.n_samples/10)))
            self.settings.setdefault('perm_range', 10)

        if self.settings['parallel_surr']:
            try:  # set default n_jobs
                if self.settings['n_jobs'] <= 0:
                    try:
                        import multiprocessing
                        n_cores = multiprocessing.cpu_count()
                        n_jobs = min(
                            n_cores + self.settings['n_jobs'] + 1, n_cores)
                        if n_jobs <= 0:
                            raise ValueError(
                                'If n_jobs has a negative value it must not be'
                                ' less than the number of CPUs present. You '
                                'have got {0} CPUs'.format(n_cores))
                    except ImportError:
                        # only warn if they tried to use something other than 1 job
                        if self.settings['n_jobs'] != 1:
                            print('multiprocessing not installed. Cannot run '
                                  'in parallel.')
                            self.settings['n_jobs'] = 1
                else:
                    self.settings.setdefault('n_jobs', 1)

            except ImportError as err:
                ex.package_missing(
                    err, 'Joblib not installed. Cannot run in parallel.')

        # Check sources and target, check links for significance
        if sources == 'all':
            sources = results.get_target_sources(
                target, fdr=self.settings['fdr_corrected'])
        elif type(sources) is list:
            sources_sign = results.get_target_sources(
                target, fdr=self.settings['fdr_corrected'])
            print('source sign')
            print(sources_sign)
            for s in sources:
                if s not in sources_sign:
                    raise RuntimeError(
                        'No significant TE between source {0} and target '
                        '{1}.'.format(s, target))

        # Get variables for requested sources.
        source_vars = []
        sourceCond = []
        for s in sources:
            temp = [i for i in results.get_single_target(
                target,
                fdr=self.settings['fdr_corrected'])['selected_vars_sources']]
            for ss in temp:
                if ss[0] == s:
                    source_vars.append(ss)
                elif ss[0] is not s and ss[0] is not source_vars:
                    sourceCond.append(ss)
            # append class tuple if class list idx_to lag give an error in indexing

        self.n_sources = len(results.get_target_sources(
            target, self.settings['fdr_corrected']))

        # Get current value and sources from previous multivar. TE analysis
        self.current_value = results.get_single_target(
            target, self.settings['fdr_corrected'])['current_value']
        self._current_value_realisations = data.get_realisations(
                                 self.current_value, [self.current_value])[0]
        self._uniq_sources = sources
        self._target = target

        # Get selected variables, convert lags in the results structure to
        # absolute indices.
        print('source var')
        print(source_vars)

        self.selected_vars_full = self._lag_to_idx(
            lag_list=(source_vars +
                      sourceCond +
                      results.get_single_target(
                          target,
                          fdr=self.settings['fdr_corrected'])['selected_vars_target']),
            current_value_sample=self.current_value[1])

        self.selected_vars_sources = self._lag_to_idx(
            lag_list=source_vars, current_value_sample=self.current_value[1])
        self.selected_vars_target = self._lag_to_idx(
            lag_list=results.get_single_target(
                target,
                fdr=self.settings['fdr_corrected'])['selected_vars_target'],
            current_value_sample=self.current_value[1])

    def spectral_surrogates(self, data_slice, scale):
        """Return spectral surrogate data for statistical testing.

        Args:
            data_slice : numpy array
                Slice of data from Data instance used, e.g., as returned by
                Data._get_data_slice()
            scale : int
                current_scale under analysis

        Returns:
            numpy array
                surrogate data with dimensions (realisations * replication)
        """
        # MODWT  (all replication in one step)
        [w_transform, approx_coeff] = modwt.modwt_C(
            data_slice, self.settings['wavelet'], self.max_scale)
        w_transform1 = np.transpose(w_transform, (1, 0, 2))
        ww = w_transform1[:, :, :]
        wav_stored = Data(ww, dim_order='psr', normalise=False)
        # wav_stored1 = Data(w_transform1, dim_order='psr', normalise=False)
        # wav_stored = Data(w_transform1, dim_order='psr', normalise=False)
        # Create surrogates by shuffling coefficients in given scale.

        if self.settings['perm_type_spec'] == 'block':
            param = {'perm_in_time': self.settings['permute_in_time_spec'],
                     'perm_type': self.settings['perm_type_spec'],
                     'block_size': self.settings['block_size_spec'],
                     'perm_range': self.settings['perm_range_spec']
                     }
        elif self.settings['perm_type_spec'] == 'circular':
            param = {'wavelet': self.settings['wavelet'],
                     'perm_in_time': self.settings['permute_in_time_spec'],
                     'perm_type': self.settings['perm_type_spec'],
                     'max_scale': self.max_scale,
                     'max_shift': self.settings['max_shift_spec']
                     }
        else:
            param = {'perm_in_time': self.settings['permute_in_time_spec'],
                     'perm_type': self.settings['perm_type_spec']
                     }

        spectral_surr = stats._generate_spectral_surrogates(
            wav_stored, scale, 1, perm_settings=param)

        wav_stored._data[scale, :, :] = spectral_surr[:, :, 0]

        merged_coeff = np.transpose(wav_stored._data, (1, 0, 2))

        # IMODWT (all replication in one step)
        rec_surrogate = modwt.imodwt_c(
            merged_coeff, approx_coeff,
            self.settings['wavelet'], self.max_scale)

        return rec_surrogate

    def GWS_surrogates(self, data_slice, scale):
        """Return modified IAAFT spectral surrogates.

        Return spectral surrogate data for statistical testing with iterative
        amplitude adjusted Fourier transform (IAAFT). It performs surrogates
        reconstruction with IAAFT amplitude correction like Keylock 2010 but
        with the following changes:

        - only a selected scale is shuffled
        - no wavelet coefficients are pinned, this is equivalent to Keylock
          code for threshold=0

        References:

        - Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for
          nonlinearity tests. Phys Rev Lett, 77(4):635.
          https://doi.org/10.1103/PhysRevLett.77.635
        - Keylock, C.J. (2010). Characterizing the structure of nonlinear
          systems using gradual wavelet reconstruction. Nonlinear Proc Geoph.
          17(6):615–632. https://doi.org/10.5194/npg-17-615-2010

        Args:
            data_slice : numpy array
                Slice of data from Data instance used, e.g., as returned by
                Data._get_data_slice()
            scale : int
                current_scale under analysis

        Returns:
            numpy array
                surrogate data with dimensions (realisations)
        """
        n_sample = data_slice.shape[0]
        [w_transform, approx_coeff] = modwt.modwt_C(
            data_slice, self.settings['wavelet'], self.max_scale)

        nsample = data_slice.shape
        numlevels = self.max_scale
        sortval2 = np.sort(data_slice)
        w_transform = np.transpose(w_transform, (1, 0, 2))
        w_transform = w_transform.reshape(self.max_scale, nsample[0])

        sizeScale = w_transform.shape

        # The threshold is based on the fact that the variance of the wavelet
        # coefficients is proportional to the Fourier energy. We place the
        # threshold at the value for the squared wavelet coefficients that
        # fixes thresh of the energy.

        reshapedata = np.sort(
            abs(w_transform.reshape((numlevels*sizeScale[1], 1))), 0)
        reshapedata = reshapedata**2
        toten = np.sum(reshapedata)
        tot = np.cumsum(reshapedata)
        tot = tot/toten

        # In the paper coeff are shuffled at each level here we shuffle only
        # at selected scale.
        currsignal = w_transform[scale, :]

        currpin = np.zeros(len(currsignal))
        meanval = np.mean(currsignal)
        stdval = np.std(currsignal)
        sortval = np.sort(currsignal-meanval)

        # Store the positions so that we can impose the pinned values below.
        cp_pos = np.where(currpin == 0)[0]
        cp_unpos = np.where(currpin != 0)[0]
        cp_unpos = []
        sortdata = currsignal[cp_pos]-meanval
        sval = np.sort(sortdata)

        fouriercoeff = abs(np.fft.ifft(currsignal - meanval)).T

        # Establish error thresholds for the amplitude and spectral parts
        # and an acceptable error.
        accerror = .01
        amperror = []
        specerror = []

        amperror.append(100)
        specerror.append(100)

        temp = []
        if np.size(cp_unpos) == 0:
            shuffind = np.argsort(np.random.rand((np.size(sval))))
            indx = np.argsort(shuffind)
            temp = sval[indx]
            y = temp
        else:
            # If there are pinned coeff it will fit and Hermitian spline
            # function (not done here).
            print('some coefficients are pinned')

        counter = 0

        # Go tho iaaft power_spectrum adjustment.
        while ((amperror[counter] > accerror) and
               (specerror[counter] > accerror)):

            # IAAFT power spectrum
            oldy = np.copy(y)
            y2 = np.fft.ifft(y)
            phase = np.angle(y2)
            y2 = fouriercoeff.T * np.exp(phase*1j)
            y = np.fft.fft(y2)
            specdiff = np.mean(np.mean(abs(np.real(y)-np.real(oldy))))
            specerror.append(specdiff/stdval)
            # specerror[counter+1]=specdiff/stdval

            # IAAFT amplitude step
            oldy = np.copy(y)
            sortdata = np.real(y[cp_pos])
            shuffind = np.argsort(sortdata)

            temp[shuffind] = sval

            y[cp_pos] = temp
            ampdiff = np.mean(np.mean(np.abs(np.real(y)-np.real(oldy))))
            amperror.append(ampdiff/stdval)

            toterror = amperror[counter+1]+specerror[counter+1]
            oldtoterr = amperror[counter]+specerror[counter]
            if (oldtoterr-toterror)/toterror < (accerror/10):
                amperror[counter+1] = -1

            counter += 1

        modwt_scale_shuff = np.real(y)+meanval
        temp_wavTranform = w_transform

        temp_wavTranform[scale, :] = modwt_scale_shuff

        temp_wavTranform = np.transpose(temp_wavTranform, (1, 0))
        temp_wavTranform = temp_wavTranform.reshape(n_sample, numlevels, 1)
        # invert step with imodwt
        newval = modwt.imodwt_c(
            temp_wavTranform, approx_coeff,
            self.settings['wavelet'], self.max_scale)

        newval = newval.reshape(n_sample)

        shuffind = np.argsort(newval, 0)

        newval[shuffind] = sortval2

        meanval = np.mean(data_slice)
        stdval = np.std(data_slice)
        sortval = np.sort(data_slice-meanval)
        fouriercoeff = abs(np.fft.ifft(data_slice - meanval)).T
        y = newval

        # additional step, repeated like above
        accerror = .01
        amperror = []
        specerror = []

        amperror.append(100)
        specerror.append(100)

        counter = 0

        # go tho iaaft power_spectrum adjustment
        while ((amperror[counter] > accerror) and
               (specerror[counter] > accerror)):

            # IAAFT power spectrum
            oldy = np.copy(y)
            y2 = np.fft.ifft(y)
            phase = np.angle(y2)
            y2 = fouriercoeff*np.exp(phase*1j)
            y = np.fft.fft(y2)
            specdiff = np.mean(np.mean(abs(np.real(y)-np.real(oldy))))
            specerror.append(specdiff/stdval)

            # IAAFT amplitude step
            oldy = np.copy(y)
            sortdata = np.real(y)
            shuffind = np.argsort(sortdata)
            y[shuffind] = sortval

            ampdiff = np.mean(np.mean(abs(np.real(y)-np.real(oldy))))
            amperror.append(ampdiff/stdval)
            # amperror[counter+1]=ampdiff/stdval

            toterror = amperror[counter+1]+specerror[counter+1]
            oldtoterr = amperror[counter]+specerror[counter]
            if (oldtoterr-toterror)/toterror < (accerror/10):

                amperror[counter+1] = -1

            counter += 1

        temp_surr = np.real(y)+meanval

        return temp_surr

    def get_surr_te(self, data, current_source, cond_set_realisations):
        """Test estimated conditional mutual information against surrogate data.

        Shuffle realisations of the source variable and re-calculate the
        multivariate transfer entropy for shuffled data.

        Args:
            data : Data instance
                raw data
            current_source : tuple
                index of current source (sample, process)
            cond_set_realisations : numpy array
                realisations of the conditioning set of the TE

        Returns:
            numpy array
                distribution of surrogate TE values
        """
        # Remember original settings.
        if self.settings['perm_type_spec'] == 'block':
            param = {'permute_in_time': self.settings['permute_in_time_spec'],
                     'perm_type': self.settings['perm_type_spec'],
                     'block_size': self.settings['block_size_spec'],
                     'perm_range': self.settings['perm_range_spec']
                     }
        elif self.settings['perm_type_spec'] == 'circular':
            param = {'permute_in_time': self.settings['permute_in_time_spec'],
                     'perm_type': self.settings['perm_type_spec'],
                     'max_shift': self.settings['max_shift_spec']
                     }
        else:
            param = {'permute_in_time': self.settings['permute_in_time_spec'],
                     'perm_type': self.settings['perm_type_spec']
                     }
        surr_realisations = stats._get_surrogates(
            data, self.current_value, current_source,
            self.settings['n_perm_spec'], param)
        surr_dist = self._cmi_estimator.estimate_parallel(
                                n_chunks=self.settings['n_perm_spec'],
                                re_use=['var2', 'conditional'],
                                var1=surr_realisations,
                                var2=self._current_value_realisations,
                                conditional=cond_set_realisations)
        return surr_dist

    def _spectral_analysis_source(self, data, results, current_scale):
        """Destroy significant source at given current_scale.

        Args:
            data : Data instance
                raw data for analysis
            results : object
                results object from Multivariate TE analysis
            current scale : int
                scale at which wavelet coefficients are shuffled

        Returns:
            object
                spectral_result
        """
        # Main algorithm.
        pvalue = []
        significance = []
        distance_delta = []
        te_full_orig = []
        te_surrogate = [[] for i in range(self.n_sources)]
        temporal_surrogate = [[] for i in range(self.n_sources)]
        temporal_rec_Surrogates = [[] for i in range(self.n_sources)]
        count = 0
        print(self._uniq_sources)
        for process in self._uniq_sources:

            print('\ntesting source {0}'.format(process))
            data_slice = data._get_data_slice(process)[0]

            # Get the conditioning set for current source to be tested and its
            # realisations (can be reused over permutations).
            cur_cond_set = [x for x in self.selected_vars_full
                            if x[0] != process]

            print('selected vars')
            print(self.selected_vars_full)

            cur_source_set = [x for x in self.selected_vars_full
                              if x[0] == process]
            # cur_source_set=[(2,22)]
            cur_cond_set_realisations = data.get_realisations(
                                self.current_value, cur_cond_set)[0]
            print('conditional_set')
            print(cur_cond_set)

            print('source_set')
            print(cur_source_set)

            print('current value')
            print(self.current_value)
            print('selected sources')
            print(self.selected_vars_sources)

            cur_source_realisationsO = data.get_realisations(
                self.current_value, cur_source_set)[0]

            # Get distribution of temporal surrogate TE values (for debugging).
            # temporal_surr = self.get_surr_te(data, cur_source_set,
            #                                  cur_cond_set_realisations)

            temporal_surr = []
            i_1 = 0
            i_2 = data.n_realisations(self.current_value)
            temp_source_realisation_perm = np.empty(
              (data.n_realisations(self.current_value) * self.settings['n_perm_spec'],
               len(cur_source_set))).astype(data.data_type)

            if self.settings['parallel_surr']:
                perm_in_time_spec = self.settings['permute_in_time_spec']

                if self.settings['surr_type'] == 'spectr':
                    print('spectral')
                    if self.settings['perm_type_spec'] == 'block':

                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale,
                                 'block_size': self.settings['block_size_spec'],
                                 'perm_range': self.settings['perm_range_spec']
                                 }

                    elif self.settings['perm_type_spec'] == 'circular':
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale,
                                 'max_shift': self.settings['max_shift_spec']
                                 }

                    else:
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale
                                 }

                    reconstructed = Parallel(
                        n_jobs=self.settings['n_jobs'],
                        verbose=self.settings['verb_parallel'])(
                            delayed(spectral_surrogates_parallel)(
                                data_slice, current_scale, param) for perm in range(0, self.settings['n_perm_spec']))

                elif self.settings['surr_type'] == 'iaaft':
                    print('iaaft')

                    if self.settings['perm_type_spec'] == 'block':
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale,
                                 'block_size': self.settings['block_size_spec'],
                                 'perm_range': self.settings['perm_range_spec']
                                 }
                    elif self.settings['perm_type_spec'] == 'circular':
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale,
                                 'max_shift': self.settings['max_shift_spec']
                                }
                    else:
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale
                                 }
                    reconstructed = [[] for i in range(data.n_replications)]
                    for rep in range(0, data.n_replications):
                        rec = Parallel(
                            n_jobs=self.settings['n_jobs'],
                            verbose=self.settings['verb_parallel'])(
                                delayed(GWS_surrogates_parallel)(
                                    data_slice[:, rep], current_scale, param) for perm in range(0, self.settings['n_perm_spec']))
                        reconstructed[rep] = rec

                    reconstructed = np.array(reconstructed)
                    reconstructed = np.transpose(reconstructed, (2, 0, 1))
                    # reconstructed replication x permutations list

                for perm in range(0, self.settings['n_perm_spec']):

                    # Get realisation for each permutation
                    print('permutation {0} of {1}'.format(
                        perm, self.settings['n_perm_spec']))

                    d_temp = cp.copy(data.data)
                    if self.settings['surr_type'] == 'spectr':
                        d_temp[process, :, :] = reconstructed[perm]

                    elif self.settings['surr_type'] == 'iaaft':
                        d_temp[process, :, :] = reconstructed[:, :, perm]

                    data_surr = Data(d_temp, 'psr', normalise=False)

                    cur_source_realisations = data_surr.get_realisations(
                                                        self.current_value,
                                                        cur_source_set)[0]

                    temp_source_realisation_perm[i_1:i_2, :] = cur_source_realisations

                    i_1 = i_2
                    i_2 += data.n_realisations(self.current_value)

                te_surr = self._cmi_estimator.estimate_parallel(
                                    n_chunks=self.settings['n_perm_spec'],
                                    re_use=['var1', 'conditional'],
                                    var1=self._current_value_realisations,
                                    var2=temp_source_realisation_perm,
                                    conditional=cur_cond_set_realisations)

            else:  # do not compute spectral surr in parallel

                for perm in range(0, self.settings['n_perm_spec']):

                    print('permutation {0} of {1}'.format(
                        perm, self.settings['n_perm_spec']))

                    if self.settings['surr_type'] == 'iaaft':
                        reconstructed = np.zeros(
                            (data.n_samples, data.n_replications))

                        for rep in range(0, data.n_replications):

                            # Create surrogate time series by reconstructing
                            # time series from shuffled coefficients for each
                            # permutation.#The Fourier Spectrum of each
                            # replication is adjusted with IAAFT algorithm (see
                            # Keylock 2010)
                            reconstructed[:, rep] = self.GWS_surrogates(
                                data_slice[:, rep], current_scale)

                    else:
                        # Create surrogate time series by reconstructing time
                        # series from shuffled coefficients for each
                        # permutation.
                        reconstructed = self.spectral_surrogates(
                            data_slice[:, :], current_scale)

                    d_temp = cp.copy(data.data)
                    d_temp[process, :, :] = reconstructed
                    data_surr = Data(d_temp, 'psr', normalise=False)

                    # Get the current source's realisations from the surrogate
                    # data object, get realisations of all other variables from
                    # the original data object.
                    cur_source_realisations = data_surr.get_realisations(
                                                        self.current_value,
                                                        cur_source_set)[0]

                    temp_source_realisation_perm[i_1:i_2, :] = cur_source_realisations
                    i_1 = i_2
                    i_2 += data.n_realisations(self.current_value)

                # Compute TE between shuffled source and current_value
                # conditional on the remaining set.
                te_surr = self._cmi_estimator.estimate_parallel(
                                    n_chunks=self.settings['n_perm_spec'],
                                    re_use=['var1', 'conditional'],
                                    var1=self._current_value_realisations,
                                    var2=temp_source_realisation_perm,
                                    conditional=cur_cond_set_realisations)

            cur_source_realisationsO = data.get_realisations(
                self.current_value, cur_source_set)[0]

            tee = self._cmi_estimator.estimate(
                    self._current_value_realisations,
                    cur_source_realisationsO,
                    cur_cond_set_realisations)

            print(tee)
            #te_all = results.get_single_target(
               # self._target, fdr=self.settings['fdr_corrected'])['te']
            #print(te_all)

            #if len(te_all) > 1:
              #  te_all = te_all[0]
            #else:
             #   print(te_all)

            te_full = tee
            te_orig_full = te_full
            # Distance between center of probability mass (te_surr) and
            # original TE_full.
            deltaO = te_orig_full - np.median(te_surr)

            # Calculate p-value for original TE against spectral surrogates.

            [sign, pval] = stats._find_pvalue(
                statistic=te_full,
                distribution=te_surr,
                alpha=self.settings['alpha_spec'],
                tail=self.settings['tail_spec'])

            te_surrogate[count].append(te_surr)
            temporal_surrogate[count].append(temporal_surrogate)
            distance_delta.append(deltaO)
            te_full_orig.append(te_orig_full)
            pvalue.append(pval)
            significance.append(sign)
            temporal_rec_Surrogates[count].append(reconstructed)
            count += 1
            #
        result_scale = {
            'te_surrogate': te_surrogate,
            'deltaO': deltaO,
            'temporal_surrogate': temporal_surr,
            'spec_pval': pvalue,
            'spec_sign': significance,
            'te_full_orig':te_full_orig,
            'selected_vars_sources': self.selected_vars_sources,
            'selected_vars_target': self.selected_vars_target,
            'source_tested': self._uniq_sources,
            'recSurrogates':temporal_rec_Surrogates}

        return result_scale

    def _spectral_analysis_target(self, data, results, current_scale):
        """Destroying significant target at given current_scale

        Args:
            data : Data instance
                raw data for analysis
            results : object
                results object from Multivariate TE analysis
            current scale : int
                scale at which wavelet coefficients are shuffled

        Returns:
            object
                spectral_result
        """
        # Main algorithm.
        pvalue = []
        significance = []
        te_full_orig = []
        te_surrogate = [[] for i in range(self.n_sources)]
        temporal_surrogate = [[] for i in range(self.n_sources)]
        temporal_rec_Surrogates = [[] for i in range(self.n_sources)]
        count = 0

        for process in self._uniq_sources:

            print('\ntesting target {0} with source {1}'.format(
                self._target, process))

            data_slice = data._get_data_slice(
                self._target)[0]  # get the data from the target

            cur_source_realisations = data.get_realisations(
                self.current_value, self.selected_vars_sources)[0]
            # Get the conditioning set for current source  and its
            # realisations (can be reused over permutations).
            cur_cond_set = [
                x for x in self.selected_vars_full if x[0] != process]
            cur_source_set = [
                x for x in self.selected_vars_full if x[0] == process]
            cur_cond_set_realisations = data.get_realisations(
                self.current_value, cur_cond_set)[0]

            print('conditional_set: {}'.format(cur_cond_set))
            print('source_set: {}'.format(cur_source_set))

            # Get distribution of temporal surrogate TE values (for debugging).
            temporal_surr = []
            i_1 = 0
            i_2 = data.n_realisations(self.current_value)
            temp_target_realisation_perm = np.empty(
              (data.n_realisations(self.current_value) * self.settings['n_perm_spec'],
               1)).astype(data.data_type)

            if self.settings['parallel_surr']:
                perm_in_time_spec = self.settings['permute_in_time_spec']
                if self.settings['surr_type'] == 'spectr':
                    if self.settings['perm_type_spec'] == 'block':
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale,
                                 'block_size': self.settings['block_size_spec'],
                                 'perm_range':self.settings['perm_range_spec']
                                 }

                    elif self.settings['perm_type_spec'] == 'circular':
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale,
                                 'max_shift': self.settings['max_shift_spec']
                                 }
                    else:
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale
                                 }

                    reconstructed = Parallel(
                        n_jobs=self.settings['n_jobs'],
                        verbose=self.settings['verb_parallel'])(
                            delayed(spectral_surrogates_parallel)(
                                data_slice, current_scale, param) for perm in range(0, self.settings['n_perm_spec']))

                elif self.settings['surr_type'] == 'iaaft':

                    if self.settings['perm_type_spec'] == 'block':
                        param = {
                            'wavelet': self.settings['wavelet'],
                            'perm_in_time': perm_in_time_spec,
                            'perm_type': self.settings['perm_type_spec'],
                            'max_scale': self.max_scale,
                            'block_size': self.settings['block_size_spec'],
                            'perm_range': self.settings['perm_range_spec']
                            }
                    elif self.settings['perm_type_spec'] == 'circular':
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale,
                                 'max_shift': self.settings['max_shift_spec']
                                 }

                    else:
                        param = {'wavelet': self.settings['wavelet'],
                                 'perm_in_time': perm_in_time_spec,
                                 'perm_type': self.settings['perm_type_spec'],
                                 'max_scale': self.max_scale
                                 }

                    reconstructed = [[] for i in range(data.n_replications)]
                    for rep in range(0, data.n_replications):

                        rec = Parallel(
                            n_jobs=self.settings['n_jobs'],
                            verbose=self.settings['verb_parallel'])(delayed(
                                GWS_surrogates_parallel)(
                                    data_slice[:, rep],
                                    current_scale, param) for perm in range(0, self.settings['n_perm_spec']))
                        reconstructed[rep] = rec

                    reconstructed = np.array(reconstructed)
                    reconstructed = np.transpose(reconstructed, (2, 0, 1))

                for perm in range(0, self.settings['n_perm_spec']):

                    print('permutation {0} of {1}'.format(
                        perm, self.settings['n_perm_spec']))

                    d_temp = cp.copy(data.data)
                    if self.settings['surr_type'] == 'spectr':
                        d_temp[self._target, :, :] = reconstructed[perm]

                    elif self.settings['surr_type'] == 'iaaft':
                        d_temp[self._target, :, :] = reconstructed[:, :, perm]

                    data_surr = Data(d_temp, 'psr', normalise=False)

                    # TODO the following can be parallelized
                    #
                    # Compute TE between shuffled source and current_value
                    # conditional on the remaining set. Get the current
                    # source's realisations from the surrogate data object, get
                    # realisations of all other variables from the original
                    # data object.
                    cur_value_target_realisations = data_surr.get_realisations(
                                                     self.current_value,
                                                     [self.current_value])[0]

                    temp_target_realisation_perm[i_1:i_2, :] = cur_value_target_realisations
                    i_1 = i_2
                    i_2 += data.n_realisations(self.current_value)

                te_surr = self._cmi_estimator.estimate_parallel(
                                    n_chunks=self.settings['n_perm_spec'],
                                    re_use=['var2', 'conditional'],
                                    var1=temp_target_realisation_perm,
                                    var2=cur_source_realisations,
                                    conditional=cur_cond_set_realisations)

            else:

                for perm in range(0, self.settings['n_perm_spec']):

                    print('permutation {0} of {1}'.format(
                        perm, self.settings['n_perm_spec']))

                    if self.settings['surr_type'] == 'iaaft':
                        reconstructed = np.zeros(
                             (data.n_samples, data.n_replications))

                        for rep in range(0, data.n_replications):
                            # Create surrogate time series by reconstructing
                            # time series from shuffled coefficients for each
                            # permutation at current scale. The Fourier
                            # Spectrum of each replication is adjusted with
                            # IAAFT algorithm
                            reconstructed[:, rep] = self.GWS_surrogates(
                                data_slice[:, rep], current_scale)
                    else:
                        # Create surrogate time series by reconstructing time
                        # series from shuffled coefficients for each
                        # permutation at current scale
                        reconstructed = self.spectral_surrogates(
                            data_slice[:, :], current_scale)

                    d_temp = cp.copy(data.data)
                    d_temp[self._target, :, :] = reconstructed
                    data_surr = Data(d_temp, 'psr', normalise=False)

                    # Get the current target's realisations from the surrogate
                    # data object while  realisations of all other variables
                    # are from the original data object.
                    cur_value_target_realisations = data_surr.get_realisations(
                                                     self.current_value,
                                                     [self.current_value])[0]

                    temp_target_realisation_perm[i_1:i_2, :] = cur_value_target_realisations

                    i_1 = i_2
                    i_2 += data.n_realisations(self.current_value)

                # Compute TE between shuffled source and current_value
                # conditional on the remaining set
                te_surr = self._cmi_estimator.estimate_parallel(
                                    n_chunks=self.settings['n_perm_spec'],
                                    re_use=['var2', 'conditional'],
                                    var1=temp_target_realisation_perm,
                                    var2=cur_source_realisations,
                                    conditional=cur_cond_set_realisations)

            cur_source_realisationsO = data.get_realisations(
                                                        self.current_value,
                                                        cur_source_set)[0]

            tee = self._cmi_estimator.estimate(
                    self._current_value_realisations,
                    cur_source_realisationsO,
                    cur_cond_set_realisations)
            print(tee)
            #te_all = results.get_single_target(self._target, fdr=self.settings['fdr_corrected'])['te']
            #print(te_all)

            # te_full = te_all[results.get_target_sources(self._target, fdr=self.settings['fdr_corrected'])]

            #if len(te_all)>1:
              #  te_all=te_all[0]

            #else:
              #  print('te')

            te_full = tee
            te_orig_full = te_full

            # Calculate p-value for original TE against spectral surrogates.
            [sign, pval] = stats._find_pvalue(
                statistic=te_full,
                distribution=te_surr,
                alpha=self.settings['alpha_spec'],
                tail=self.settings['tail_spec'])

            te_surrogate[count].append(te_surr)
            temporal_surrogate[count].append(temporal_surrogate)
            te_full_orig.append(te_orig_full)
            pvalue.append(pval)
            significance.append(sign)
            temporal_rec_Surrogates[count].append(reconstructed)
            count += 1

        result_scale = {
            'te_surrogate': te_surrogate,
            'deltaO': [],
            'temporal_surrogate': temporal_surr,
            'spec_pval': pvalue,
            'spec_sign': significance,
            'te_full_orig': te_full_orig,
            'selected_vars_sources': self.selected_vars_sources,
            'selected_vars_target': self.selected_vars_target,
            'source_tested': self._uniq_sources,
            'recSurrogates': temporal_rec_Surrogates}
        return result_scale

    def _spectral_analysis_SOS_T(self, data, results_spectral, scale_source,
                                 scale_target):
        """Perform post-hoc test of source-target interaction.

        This function perform a post-hoc analysis between source-target
        interaction (PID) to estimate if the source with the maximal distance
        to the TE_full origin is the only source of interaction with the
        target, since many-to-many relationships source and sender are possible
        when there is information transfer.
        """
        print('computing distance TE destroyed target with TE_ original')
        print(scale_target)
        print(scale_source)
        print(self._target)

        # Main algorithm.
        pvalue = []
        significance = []
        delta_surrogate = []
        data_slice = data._get_data_slice(self._target)[0] # get target data
        print(self.selected_vars_sources)

        cur_source_realisations = data.get_realisations(
            self.current_value, self.selected_vars_sources)[0]
        # Get the conditioning set for current source  and its
        # realisations (can be reused over permutations).
        print(self._uniq_sources)
        cur_cond_set = [x for x in self.selected_vars_full
                        if x[0] != self._uniq_sources[0]]
        cur_source_set = [x for x in self.selected_vars_full
                          if x[0] == self._uniq_sources[0]]

        cur_cond_set_realisations = data.get_realisations(
                            self.current_value, cur_cond_set)[0]

        print(cur_cond_set)

        # Get distribution of temporal surrogate TE values (for debugging).
        # temporal_surr = self.get_surr_te(data, cur_source_set,
        #                                  cur_cond_set_realisations)

        if self.settings['surr_type'] == 'spectr':
            if self.settings['perm_type_spec'] == 'block':
                param = {
                    'wavelet': self.settings['wavelet'],
                    'perm_in_time': self.settings['permute_in_time_spec'],
                    'perm_type': self.settings['perm_type_spec'],
                    'max_scale': self.max_scale,
                    'block_size': self.settings['block_size_spec'],
                    'perm_range': self.settings['perm_range_spec']
                }
            print('modwt')
            reconstructed = Parallel(
                n_jobs=self.settings['n_jobs'],
                verbose=self.settings['verb_parallel'])(delayed(
                    spectral_surrogates_parallel)(data_slice, scale_target, param) for perm in range(0, self.settings['n_perm_spec']))

        elif self.settings['surr_type'] == 'iaaft':

            if self.settings['perm_type_spec'] == 'block':
                param = {'wavelet': self.settings['wavelet'],
                         'perm_in_time': self.settings['permute_in_time_spec'],
                         'perm_type': self.settings['perm_type_spec'],
                         'max_scale': self.max_scale,
                         'block_size': self.settings['block_size_spec'],
                         'perm_range': self.settings['perm_range_spec']
                         }

            elif self.settings['perm_type_spec'] == 'circular':
                param = {'wavelet': self.settings['wavelet'],
                         'perm_in_time': self.settings['permute_in_time_spec'],
                         'perm_type': self.settings['perm_type_spec'],
                         'max_scale': self.max_scale,
                         'max_shift': self.settings['max_shift_spec']
                         }
            else:
                param = {'wavelet': self.settings['wavelet'],
                         'perm_in_time': self.settings['permute_in_time_spec'],
                         'perm_type': self.settings['perm_type_spec'],
                         'max_scale': self.max_scale
                         }

            reconstructed = [[] for i in range(data.n_replications)]
            for rep in range(0, data.n_replications):
                rec = Parallel(n_jobs=self.settings['n_jobs'],
                               verbose=self.settings['verb_parallel'])(delayed(
                                    GWS_surrogates_parallel)(
                                        data_slice[:, rep],
                                        scale_target,
                                        param) for perm in range(0, self.settings['n_perm_spec']))
                reconstructed[rep] = rec

            reconstructed = np.array(reconstructed)
            reconstructed = np.transpose(reconstructed, (2, 0, 1))

        # store te_target
        delta_S_T = np.zeros(self.settings['n_perm_spec'])
        for n_perm in range(0, self.settings['n_perm_spec']):
            print('permutation {0} of {1}'.format(
                n_perm, self.settings['n_perm_spec']))

            d_temp = cp.copy(data.data)
            print(self._target)
            if self.settings['surr_type'] == 'spectr':
                d_temp[self._target, :, :] = reconstructed[n_perm]

            elif self.settings['surr_type'] == 'iaaft':
                d_temp[self._target, :, :] = reconstructed[:, :, n_perm]

            data_surr = Data(d_temp, 'psr', normalise=False)

            # TODO the following can be parallelized
            # Compute TE between shuffled source and current_value
            # conditional on the remaining set. Get the current source's
            # realisations from the surrogate data object, get realisations
            # of all other variables from the original data object.
            cur_value_target_realisationsD = data_surr.get_realisations(
                                             self.current_value,
                                             [self.current_value])[0]

            te_surr_target = self._cmi_estimator.estimate(
                                        var1=cur_value_target_realisationsD,
                                        var2=cur_source_realisations,
                                        conditional=cur_cond_set_realisations)

            data_slice_source = data._get_data_slice(self._uniq_sources[0])[0]

            # Get the conditioning set for current source to be tested and its
            # realisations (can be reused over permutations).
            cur_cond_set = [x for x in self.selected_vars_full
                            if x[0] != self._uniq_sources[0]]
            cur_source_set = [x for x in self.selected_vars_full
                              if x[0] == self._uniq_sources[0]]
            cur_cond_set_realisations = data.get_realisations(
                                self.current_value, cur_cond_set)[0]

            if self.settings['surr_type'] == 'spectr':

                if self.settings['perm_type_spec'] == 'block':
                    param = {'wavelet': self.settings['wavelet'],
                             'perm_in_time': self.settings['permute_in_time_spec'],
                             'perm_type': self.settings['perm_type_spec'],
                             'max_scale': self.max_scale,
                             'block_size': self.settings['block_size_spec'],
                             'perm_range': self.settings['perm_range_spec']
                             }
                reconstructedS = Parallel(
                    n_jobs=self.settings['n_jobs'],
                    verbose=self.settings['verb_parallel'])(delayed(
                        spectral_surrogates_parallel)(data_slice_source, scale_source, param) for perm in range(0, self.settings['n_perm_spec']))

            elif self.settings['surr_type'] == 'iaaft':
                if self.settings['perm_type_spec'] == 'block':
                    param = {
                        'wavelet': self.settings['wavelet'],
                        'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],
                        'max_scale': self.max_scale,
                        'block_size': self.settings['block_size_spec'],
                        'perm_range': self.settings['perm_range_spec']
                        }

                elif self.settings['perm_type_spec'] == 'circular':
                    param = {
                        'wavelet': self.settings['wavelet'],
                        'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],
                        'max_scale': self.max_scale,
                        'max_shift': self.settings['max_shift_spec']
                        }
                else:
                    param = {
                        'wavelet': self.settings['wavelet'],
                        'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],
                        'max_scale': self.max_scale
                        }

                reconstructedS = [[] for i in range(data.n_replications)]
                for rep in range(0, data.n_replications):
                    rec = Parallel(n_jobs=self.settings['n_jobs'], verbose=self.settings['verb_parallel'])(delayed(GWS_surrogates_parallel)(
                        data_slice_source[:, rep], scale_source, param) for perm in range(0, self.settings['n_perm_spec']))
                    reconstructedS[rep] = rec
                reconstructedS = np.array(reconstructedS)
                reconstructedS = np.transpose(reconstructedS, (2, 0, 1))

            te_surr_source = np.zeros(self.settings['n_perm_spec'])
            for k_perm in range(0, self.settings['n_perm_spec']):

                # Get realisation for each permutation
                print('permutation {0} of {1}'.format(
                    k_perm, self.settings['n_perm_spec']))

                # d_tempS = cp.copy(data.data)
                if self.settings['surr_type'] == 'spectr':
                    d_temp[self._uniq_sources[0], :, :] = reconstructedS[k_perm]
                elif self.settings['surr_type'] == 'iaaft':
                    d_temp[self._uniq_sources[0], :, :] = reconstructedS[:, :, k_perm]

                data_surrS = Data( d_temp, 'psr', normalise=False)

                cur_source_realisations_destroyed = data_surrS.get_realisations(
                    self.current_value, cur_source_set)[0]
                cur_value_target_realisationsDD = data_surrS.get_realisations(
                    self.current_value, [self.current_value])[0]

                te_surr_source[k_perm] = self._cmi_estimator.estimate(
                                    var1=cur_value_target_realisationsDD,
                                    var2=cur_source_realisations_destroyed,
                                    conditional=cur_cond_set_realisations)

            delta_s_prime = np.median(te_surr_source)
            delta_S_T[n_perm] = te_surr_target-delta_s_prime
            print(delta_S_T)

        # Check if te_original soource  is in the extreme 5% of the ditribution
        # dist_Te_targetvsSource.
        delta_orig = self.settings['delta']
        [sign, pval] = stats._find_pvalue(statistic=delta_orig,
                                          distribution=delta_S_T,
                                          alpha=self.settings['alpha_spec'],
                                          tail=self.settings['tail_spec'])
        delta_surrogate.append(delta_S_T)
        pvalue.append(pval)
        significance.append(sign)
        result_scale = {'delta_surrogate': delta_surrogate,
                        'spec_pval': pvalue,
                        'spec_sign': significance,
                        'delta': delta_orig
                        }
        return result_scale

    def _spectral_analysis_SOS_S(self, data, results_spectral, scale_source,
                                 scale_target):
        """Perform post-hoc analysis of source-target interaction.

        This function perform a post-hoc analysis between source-target
        interaction (PID) to estimate if the source with the maximal distance
        to the TE_full origin is the only source of interaction with the
        target, since many-to-many relationships source and sender are possible
        when there is information transfer.
        """
        print('computing distance TE destroyed target with TE_ original')

        # Main algorithm.
        pvalue = []
        significance = []
        delta_surrogate = []
        # Destroy source and then destroy target
        print(self._uniq_sources)
        print(scale_target)
        data_slice_source = data._get_data_slice(self._uniq_sources[0])[0]

        # Get the conditioning set for current source to be tested and its
        # realisations (can be reused over permutations).
        cur_cond_set = [x for x in self.selected_vars_full
                        if x[0] != self._uniq_sources[0]]
        cur_source_set = [x for x in self.selected_vars_full
                          if x[0] == self._uniq_sources[0]]
        cur_cond_set_realisations = data.get_realisations(
                            self.current_value, cur_cond_set)[0]

        cur_value_target_realisations = data.get_realisations(
                                            self.current_value,
                                            [self.current_value])[0]

        # Get distribution of temporal surrogate TE values (for debugging).
        # temporal_surr = self.get_surr_te(data, cur_source_set,
        #                                  cur_cond_set_realisations)

        if self.settings['surr_type'] == 'spectr':
            if self.settings['perm_type_spec'] == 'block':
                param = {'wavelet': self.settings['wavelet'],
                         'perm_in_time': self.settings['permute_in_time_spec'],
                         'perm_type': self.settings['perm_type_spec'],
                         'max_scale': self.max_scale,
                         'block_size': self.settings['block_size_spec'],
                         'perm_range': self.settings['perm_range_spec']
                         }

            print('modwt')
            reconstructed = Parallel(n_jobs=self.settings['n_jobs'], verbose=self.settings['verb_parallel'])(delayed(
                spectral_surrogates_parallel)(data_slice_source, scale_source, param) for perm in range(0, self.settings['n_perm_spec']))

        elif self.settings['surr_type'] == 'iaaft':

            if self.settings['perm_type_spec'] == 'block':
                param = {'wavelet': self.settings['wavelet'],
                         'perm_in_time': self.settings['permute_in_time_spec'],
                         'perm_type': self.settings['perm_type_spec'],
                         'max_scale': self.max_scale,
                         'block_size': self.settings['block_size_spec'],
                         'perm_range': self.settings['perm_range_spec']
                         }

            elif self.settings['perm_type_spec'] == 'circular':
                param = {'wavelet': self.settings['wavelet'],
                         'perm_in_time': self.settings['permute_in_time_spec'],
                         'perm_type': self.settings['perm_type_spec'],
                         'max_scale': self.max_scale,
                         'max_shift': self.settings['max_shift_spec']
                         }

            else:
                param = {'wavelet': self.settings['wavelet'],
                         'perm_in_time': self.settings['permute_in_time_spec'],
                         'perm_type': self.settings['perm_type_spec'],
                         'max_scale': self.max_scale
                         }

            reconstructed = [[] for i in range(data.n_replications)]
            for rep in range(0, data.n_replications):
                rec = Parallel(n_jobs=self.settings['n_jobs'], verbose=self.settings['verb_parallel'])(delayed(GWS_surrogates_parallel)(
                    data_slice_source[:, rep], scale_source, param) for perm in range(0, self.settings['n_perm_spec']))
                reconstructed[rep] = rec

            reconstructed = np.array(reconstructed)
            reconstructed = np.transpose(reconstructed, (2, 0, 1))

        # Store te_target
        delta_S_T = np.zeros(self.settings['n_perm_spec'])
        for n_perm in range(0, self.settings['n_perm_spec']):
            print('permutation {0} of {1}'.format(
                n_perm, self.settings['n_perm_spec']))

            d_temp = cp.copy(data.data)
            print(self._target)
            if self.settings['surr_type'] == 'spectr':
                d_temp[self._uniq_sources[0], :, :] = reconstructed[n_perm]

            elif self.settings['surr_type'] == 'iaaft':
                d_temp[self._uniq_sources[0], :, :] = reconstructed[:, :, n_perm]

            data_surr = Data(d_temp, 'psr', normalise=False)

            # TODO the following can be parallelized
            # Compute TE between shuffled source and current_value
            # conditional on the remaining set. Get the current source's
            # realisations from the surrogate data object, get realisations
            # of all other variables from the original data object.

            cur_source_realisations_destroyed = data_surr.get_realisations(
                                                        self.current_value,
                                                        cur_source_set)[0]
            te_surr_source = self._cmi_estimator.estimate(
                                var1=cur_value_target_realisations,
                                var2=cur_source_realisations_destroyed,
                                conditional=cur_cond_set_realisations)

            data_slice = data._get_data_slice(self._target)[0]  # get target data

            # Get the conditioning set for current source  and its
            # realisations (can be reused over permutations).
            print(self._uniq_sources)
            cur_cond_set = [x for x in self.selected_vars_full
                            if x[0] != self._uniq_sources[0]]
            cur_source_set = [x for x in self.selected_vars_full
                              if x[0] == self._uniq_sources[0]]
            cur_cond_set_realisations = data.get_realisations(
                                self.current_value, cur_cond_set)[0]

            if self.settings['surr_type'] == 'spectr':
                if self.settings['perm_type_spec'] == 'block':
                    param = {
                        'wavelet': self.settings['wavelet'],
                        'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],
                        'max_scale': self.max_scale,
                        'block_size': self.settings['block_size_spec'],
                        'perm_range': self.settings['perm_range_spec']
                        }
                reconstructedS = Parallel(
                    n_jobs=self.settings['n_jobs'],
                    verbose=self.settings['verb_parallel'])(
                        delayed(spectral_surrogates_parallel)(
                            data_slice, scale_target, param) for perm in range(0, self.settings['n_perm_spec']))

            elif self.settings['surr_type'] == 'iaaft':

                if self.settings['perm_type_spec'] == 'block':
                    param = {
                        'wavelet': self.settings['wavelet'],
                        'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],
                        'max_scale': self.max_scale,
                        'block_size': self.settings['block_size_spec'],
                        'perm_range': self.settings['perm_range_spec']
                        }

                elif self.settings['perm_type_spec'] == 'circular':
                    param = {
                        'wavelet': self.settings['wavelet'],
                        'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],
                        'max_scale': self.max_scale,
                        'max_shift': self.settings['max_shift_spec']
                        }

                else:
                    param = {
                        'wavelet': self.settings['wavelet'],
                        'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],
                        'max_scale': self.max_scale
                        }

                reconstructedS = [[] for i in range(data.n_replications)]
                for rep in range(data.n_replications):
                    rec = Parallel(
                        n_jobs=self.settings['n_jobs'],
                        verbose=self.settings['verb_parallel'])(
                            delayed(GWS_surrogates_parallel)(
                                data_slice[:, rep], scale_target, param) for perm in range(0, self.settings['n_perm_spec']))
                    reconstructedS[rep] = rec

                reconstructedS = np.array(reconstructedS)
                reconstructedS = np.transpose(reconstructedS, (2, 0, 1))

            te_surr_target = np.zeros(self.settings['n_perm_spec'])
            for k_perm in range(0, self.settings['n_perm_spec']):

                # Get realisation for each permutation
                print('permutation {0} of {1}'.format(
                    k_perm, self.settings['n_perm_spec']))

                if self.settings['surr_type'] == 'spectr':
                    d_temp[self._target, :, :] = reconstructedS[k_perm]
                elif self.settings['surr_type'] == 'iaaft':
                    d_temp[self._target, :, :] = reconstructedS[:, :, k_perm]

                data_surrS = Data(d_temp, 'psr', normalise=False)
                cur_value_target_realisations_destroyed = data_surrS.get_realisations(
                                            self.current_value,
                                            [self.current_value])[0]
                cur_source_realisations_destroyedS = data_surrS.get_realisations(
                                                    self.current_value,
                                                    cur_source_set)[0]
                te_surr_target[k_perm] = self._cmi_estimator.estimate(
                    var1=cur_value_target_realisations_destroyed,
                    var2=cur_source_realisations_destroyedS,
                    conditional=cur_cond_set_realisations)

            delta_s_prime = np.median(te_surr_target)
            delta_S_T[n_perm] = delta_s_prime-te_surr_source
            print(delta_S_T[n_perm])

        # Check if te_original soource  is in the extreme 5% of the
        # ditribution dist_Te_targetvsSource
        delta_orig = self.settings['delta']
        [sign, pval] = stats._find_pvalue(
            statistic=delta_orig,
            distribution=delta_S_T,
            alpha=self.settings['alpha_spec'],
            tail=self.settings['tail_spec'])
        delta_surrogate.append(delta_S_T)
        pvalue.append(pval)
        significance.append(sign)
        result_scale = {
            'delta_surrogate': delta_surrogate,
            'spec_pval': pvalue,
            'spec_sign': significance,
            'delta': delta_orig
            }
        return result_scale


def spectral_surrogates_parallel(data_slice, scale, param):
    """Return spectral surrogate data for statistical testing.

    Args:
        data_slice : numpy array
            Slice of data from Data instance used, e.g., as returned by
            Data._get_data_slice()
        scale : int
            current_scale under analysis
        param : dict
            Surrogate parameters

    Returns:
        numpy array
            surrogate data with dimensions (realisations * replication)
    """
    # all replication in one steps
    [w_transform, approx_coeff] = modwt.modwt_C(
        data_slice, param['wavelet'], param['max_scale'])
    w_transform1 = np.transpose(w_transform, (1, 0, 2))
    wav_stored = Data(w_transform1, dim_order='psr', normalise=False)
    # Create surrogates by shuffling coefficients in given scale.
    spectral_surr = _generate_spectral_surrogates(
        wav_stored, scale, n_perm=1, perm_settings=param)
    wav_stored._data[scale, :, :] = spectral_surr[:, :, 0]
    merged_coeff = np.transpose(wav_stored._data, (1, 0, 2))
    rec_surrogate = modwt.imodwt_c(
        merged_coeff, approx_coeff, param['wavelet'],param['max_scale'])
    return rec_surrogate


def GWS_surrogates_parallel(data_slice, scale, param):
    """Return modified IAAFT spectral surrogates.

    Return spectral surrogate data for statistical testing with Iterative
    Amplitude Adjustment (IAAFT). It performs surrgates reconstruction whith
    IAAFT amplitude correction like Keylock 2010 but with the following
    changes:

    - only a selected scale is shuffled
    - no wavelet coefficients are pinned, this is equivalent to Keylock code
      for threshold=0

    References:

    - Keylock, C.J. (2010). Characterizing the structure of nonlinear
        systems using gradual wavelet reconstruction. Nonlinear Proc Geoph.
        17(6):615–632. https://doi.org/10.5194/npg-17-615-2010

    Args:
        data_slice : numpy array
            Slice of data from Data instance used, e.g., as returned by
            Data._get_data_slice()
        scale : int
            current_scale under analysis

    Returns:
        numpy array
            surrogate data with dimensions (realisations)
    """
    n_sample = data_slice.shape[0]
    [w_transform, approx_coeff] = modwt.modwt_C(
        data_slice, param['wavelet'], param['max_scale'])
    nsample = data_slice.shape
    numlevels = param['max_scale']
    sortval2 = np.sort(data_slice)
    w_transform = np.transpose(w_transform, (1, 0, 2))
    w_transform = w_transform.reshape(param['max_scale'], nsample[0])

    sizeScale = w_transform.shape

    # The threshold is based on the fact that the variance of the wavelet
    # coefficients is proportional to the Fourier energy. We place the
    # threshold at the value for the squared wavelet coefficients that fixes
    # thresh of the energy.

    reshapedata = np.sort(
        abs(w_transform.reshape((numlevels*sizeScale[1], 1))), 0)
    reshapedata = reshapedata**2
    toten = np.sum(reshapedata)
    tot = np.cumsum(reshapedata)
    tot = tot/toten

    # In the paper coeff are shuffled at each level here we shuffle only at
    # selected scale.
    currsignal = w_transform[scale, :]

    currpin = np.zeros(len(currsignal))
    meanval = np.mean(currsignal)
    stdval = np.std(currsignal)
    sortval = np.sort(currsignal-meanval)

    # Store the positions so that we can impose the pinned values below
    cp_pos = np.where(currpin == 0)[0]
    cp_unpos = np.where(currpin != 0)[0]
    cp_unpos = []
    sortdata = currsignal[cp_pos]-meanval
    sval = np.sort(sortdata)

    fouriercoeff = abs(np.fft.ifft(currsignal - meanval)).T

    # Establish error thresholds for the amplitude and spectral parts
    # and an acceptable error.
    accerror = .001
    amperror = []
    specerror = []

    amperror.append(100)
    specerror.append(100)

    temp = []
    if np.size(cp_unpos) == 0:
        shuffind = np.argsort(np.random.rand((np.size(sval))))
        indx = np.argsort(shuffind)
        temp = sval[indx]
        y = temp
    else:
        # If there are pinned coeff it will fit and hermitian spline function
        # (not done here).
        print('some coefficients are pinned')

    counter = 0

    # go tho iaaft power_spectrum adjustment
    while (amperror[counter] > accerror) and (specerror[counter] > accerror):

        # IAAFT power spectrum
        oldy = np.copy(y)
        y2 = np.fft.ifft(y)
        phase = np.angle(y2)
        y2 = fouriercoeff.T * np.exp(phase*1j)
        y = np.fft.fft(y2)
        specdiff = np.mean(np.mean(abs(np.real(y)-np.real(oldy))))
        specerror.append(specdiff/stdval)

        # IAAFT amplitude step
        oldy = np.copy(y)
        sortdata = np.real(y[cp_pos])
        shuffind = np.argsort(sortdata)

        temp[shuffind] = sval

        y[cp_pos] = temp

        # y[cp_unpos]=currsignal[cp_unpos]-meanval
        ampdiff = np.mean(np.mean(np.abs(np.real(y)-np.real(oldy))))
        amperror.append(ampdiff/stdval)
        # amperror[counter+1]=ampdiff/stdval

        toterror = amperror[counter+1]+specerror[counter+1]
        oldtoterr = amperror[counter]+specerror[counter]
        if (oldtoterr-toterror)/toterror < (accerror/10):
            amperror[counter+1] = -1
        counter += 1

    modwt_scale_shuff = np.real(y)+meanval
    temp_wavTranform = w_transform

    temp_wavTranform[scale, :] = modwt_scale_shuff

    temp_wavTranform = np.transpose(temp_wavTranform, (1, 0))
    temp_wavTranform = temp_wavTranform.reshape(n_sample, numlevels, 1)

    # invert step with imodwt
    newval = modwt.imodwt_c(temp_wavTranform, approx_coeff,
                            param['wavelet'], param['max_scale'])

    newval = newval.reshape(n_sample)
    # err=np.linalg.norm(np.abs(data_slice-newval))
    # print(err)
    shuffind = np.argsort(newval, 0)
    newval[shuffind] = sortval2

    meanval = np.mean(data_slice)
    stdval = np.std(data_slice)
    sortval = np.sort(data_slice-meanval)
    fouriercoeff = abs(np.fft.ifft(data_slice - meanval)).T
    y = newval

    # additional step, repeated like above
    accerror = .001
    amperror = []
    specerror = []

    amperror.append(100)
    specerror.append(100)

    counter = 0

    # go tho iaaft power_spectrum adjustment
    while (amperror[counter] > accerror) and (specerror[counter] > accerror):

        # IAAFT power spectrum
        oldy = np.copy(y)
        y2 = np.fft.ifft(y)
        phase = np.angle(y2)
        y2 = fouriercoeff*np.exp(phase*1j)
        y = np.fft.fft(y2)
        specdiff = np.mean(np.mean(abs(np.real(y)-np.real(oldy))))
        specerror.append(specdiff/stdval)

        # IAAFT amplitude step
        sortdata = None
        shuffind = None
        oldy = np.copy(y)
        sortdata = np.real(y)
        shuffind = np.argsort(sortdata)
        y[shuffind] = sortval

        ampdiff = np.mean(np.mean(abs(np.real(y)-np.real(oldy))))
        amperror.append(ampdiff/stdval)
        # amperror[counter+1]=ampdiff/stdval

        toterror = amperror[counter+1]+specerror[counter+1]
        oldtoterr = amperror[counter]+specerror[counter]
        if (oldtoterr-toterror)/toterror < (accerror/10):

            amperror[counter+1] = -1

        counter += 1
    amperror = None
    specerror = None
    currsignal = None
    temp_surr = np.real(y)+meanval

    return temp_surr
