"""Parent class for network inference and network comparison.
"""
import ast
import copy as cp
import itertools as it
import os.path
from datetime import datetime
from pprint import pprint
from shutil import copyfile

import numpy as np

from . import idtxl_io as io
from . import idtxl_utils as utils
from .estimator import get_estimator


class NetworkAnalysis:
    """Provide an analysis setup for network inference or comparison.

    The class provides routines to check user input and set defaults.
    """

    def __init__(self):
        self.settings = {}
        self.target = None
        self.current_value = None
        self.selected_vars_full = []
        self.selected_vars_sources = []
        self.selected_vars_target = []
        self._current_value_realisations = None
        self._selected_vars_realisations = None
        self._min_stats_surr_table = None

    @property
    def current_value(self):
        """Get index of the current_value."""
        return self._current_value

    @current_value.setter
    def current_value(self, idx):
        if (idx is not None) and (type(idx) is not tuple):
            raise TypeError(
                (
                    "The current value should be a tuple (index "
                    + "process, index sample)."
                )
            )
        self._current_value = idx

    @property
    def _current_value_realisations(self):
        """Get realisations of the current_value."""
        if self.__current_value_realisations is None:
            print("Attribute has not been set yet.")
        if type(self.__current_value_realisations) is tuple:
            raise TypeError("something went wrong")
        return self.__current_value_realisations

    @_current_value_realisations.setter
    def _current_value_realisations(self, realisations):
        self.__current_value_realisations = realisations

    @_current_value_realisations.deleter
    def _current_value_realisations(self):
        del self.__current_value_realisations

    @property
    def selected_vars_full(self):
        """List of indices of the full conditional set."""
        if self._selected_vars_full is None:
            print("Attribute has not been set yet.")
        return self._selected_vars_full

    @selected_vars_full.setter
    def selected_vars_full(self, idx_list):
        if type(idx_list) is not list and (type(idx_list[0]) is not tuple):
            raise TypeError(
                ("Expected a list of tuples (index process, " + "index sample).")
            )
        self._selected_vars_full = idx_list

    @property
    def selected_vars_target(self):
        """List of indices of target samples in the conditional set."""
        if self._selected_vars_target is None:
            print("Attribute has not been set yet.")
        return self._selected_vars_target

    @selected_vars_target.setter
    def selected_vars_target(self, idx_list):
        if idx_list is not None and type(idx_list) is not list:
            raise TypeError(
                ("Expected a list of tuples (index process, " + "index sample).")
            )
        self._selected_vars_target = idx_list

    @property
    def selected_vars_sources(self):
        """List of indices of source samples in the conditional set."""
        if self._selected_vars_sources is None:
            print("Attribute has not been set yet.")
        return self._selected_vars_sources

    @selected_vars_sources.setter
    def selected_vars_sources(self, idx_list):
        if idx_list is not None and type(idx_list) is not list:
            raise TypeError(
                ("Expected a list of tuples (index process, " + "index sample).")
            )
        self._selected_vars_sources = idx_list

    @property
    def _selected_vars_realisations(self):
        """Get realisations of the full conditional set."""
        return self.__selected_vars_realisations

    @_selected_vars_realisations.setter
    def _selected_vars_realisations(self, realisations):
        self.__selected_vars_realisations = realisations

    def _idx_to_lag(self, idx_list, current_value_sample=None):
        """Change sample indices to lags for each sample in the list."""
        if current_value_sample is None:
            try:
                current_value_sample = self.current_value[1]
            except (AttributeError, TypeError):
                raise AttributeError("Current value not set.")

        lag_list = cp.copy(idx_list)
        for c in idx_list:
            if c[1] > current_value_sample:
                raise IndexError("Sample time index larger than current " "value.")
            lag_list[idx_list.index(c)] = (c[0], current_value_sample - c[1])
        return lag_list

    def _lag_to_idx(self, lag_list, current_value_sample=None):
        """Change sample lags to indices for each sample in the list."""
        if current_value_sample is None:
            try:
                current_value_sample = self.current_value[1]
            except (AttributeError, TypeError):
                raise AttributeError("Current value not set.")

        idx_list = cp.copy(lag_list)
        for c in lag_list:
            if c[1] > current_value_sample:
                raise IndexError("Sample lag larger than current value.")
            idx_list[lag_list.index(c)] = (c[0], current_value_sample - c[1])
        return idx_list

    def _set_cmi_estimator(self):
        """Check and set requested CMI estimator."""
        # Set CMI estimator. Check if the user requested the estimation of
        # local values. If so, initialise a local estimator additionally to the
        # average estimator. Internally, the average estimator is used for
        # building the non-uniform embedding, etc. The local estimator is used
        # to estimate single-link MI/TE or single-process AIS in the end.
        assert "cmi_estimator" in self.settings, "Estimator was not specified!"

        if self.settings["local_values"]:
            self.settings["local_values"] = False
            self._cmi_estimator = get_estimator(
                self.settings["cmi_estimator"], self.settings
            )
            self.settings["local_values"] = True
            self._cmi_estimator_local = get_estimator(
                self.settings["cmi_estimator"], self.settings
            )
        else:
            self._cmi_estimator = get_estimator(
                self.settings["cmi_estimator"], self.settings
            )

    def _separate_realisations(self, idx_full, idx_single):
        """Separate single index realisations from a set of realisations.

        Return the realisations of a single index and the realisations of the
        remaining set of indices. The function takes realisations from the
        array in self._selected_vars_realisations. This allows to reuse the
        collected realisations when pruning the conditional set after
        candidates have been included.

        Args:
            idx_full : list of tuples
                indices indicating the full set
            idx_single : tuple
                index to be removed

        Returns:
            numpy array
                realisations of the set without the single index
            numpy array
                realisations of the variable at the single index
        """
        # Get indices of the remaining variables.
        idx_remaining = cp.copy(idx_full)
        idx_remaining.pop(idx_remaining.index(idx_single))

        # Find the indices of the columns with the realisations of the
        # requested variables (the single one to be removed and the remaining
        # variables).
        array_col_single = self.selected_vars_full.index(idx_single)
        array_col_remain = np.zeros(len(idx_remaining)).astype(int)
        for i, idx in enumerate(idx_remaining):
            array_col_remain[i] = self.selected_vars_full.index(idx)

        # Get realisations of the single and remaining variables.
        real_single = np.expand_dims(
            self._selected_vars_realisations[:, array_col_single], axis=1
        )
        if len(idx_full) == 1:
            # If no realiastions remain, set variable to None instead of and
            # empty array so the JIDT estimator doesn't break
            real_remain = None
        else:
            real_remain = self._selected_vars_realisations[:, array_col_remain]

        return real_remain, real_single

    def _define_candidates(self, processes, samples):
        """Build a list of candidate indices.

        Build a list of candidate indices. Note that variables that were
        manually added to the conditioning set via the 'add_conditionals'
        setting are removed from the candidate set if both sets are not
        disjoint.

        Args:
            processes : list of int
                process indices
            samples: list of int
                sample indices

        Returns:
            a list of tuples, where each tuple holds the index of one
            candidate and has the form (process index, sample index), indices
            are absolute values with respect to some data array.
        """
        candidate_set = self._build_variable_list(processes, samples)
        # Remove candidates that were already manullay added to the
        # conditioning set via the 'add_conditionals' setting. Otherwise the
        # candidates get tested in the inclusion step.
        candidate_set = self._remove_forced_conditionals(candidate_set)
        return candidate_set

    def _build_variable_list(self, processes, samples):
        """Build a list of variable tuples with (process index, sample index).

        Args:
            processes : list of int
                process indices
            samples: list of int
                sample indices

        Returns:
            a list of variable tuples
        """
        var_list = []
        for idx in it.product(processes, samples):
            var_list.append(idx)
        return var_list

    def _remove_forced_conditionals(self, candidate_set):
        """Remove enforced conditioning variables from candidate set."""
        if self.settings["add_conditionals"] is not None:
            cond = self.settings["add_conditionals"]
            if type(cond) is tuple:  # easily add single variable
                cond = [cond]
            elif type(cond) is dict:  # add conditioning variables per target
                try:
                    cond = cond[self.target]
                except KeyError:
                    return  # no additional variables for the current target
            cond_idx = self._lag_to_idx(cond)
            candidate_set = list(set(candidate_set).difference(set(cond_idx)))
        return candidate_set

    def _append_selected_vars_idx(self, idx):
        """Append indices of conditionals to existing list.

        Args:
            idx : list of tuples
                indices of selected variables, where each entry is a tuple
                (idx process, idx sample), where indices are absolute values
                with respect to entries in a data array
        """
        if self.selected_vars_full is None:
            self.selected_vars_full = idx
        else:
            for i in idx:
                self.selected_vars_full.append(i)
        # separate indices into source and target indices
        for i in idx:
            if i[0] == self.target:
                self.selected_vars_target.append(i)
            else:
                self.selected_vars_sources.append(i)

    def _append_selected_vars(self, data, idx):
        """Append indices and realisation of selected variables.

        Args:
            data : Data instance
            idx : list of tuples
                indices of selected variables, where each entry is a tuple
                (idx process, idx sample), where indices are absolute values
                with respect to entries in a data array
        """
        self._append_selected_vars_idx(idx)
        self._selected_vars_realisations = data.get_realisations(self._current_value, self.selected_vars_full)

    def _remove_selected_var(self, data, idx):
        """Remove a single selected variable and its realisations."""
        self.selected_vars_full.pop(self.selected_vars_full.index(idx))
        self._selected_vars_realisations = data.get_realisations(self._current_value, self.selected_vars_full)
        
        if idx[0] == self.target:
            self.selected_vars_target.pop(self.selected_vars_target.index(idx))
        else:
            self.selected_vars_sources.pop(self.selected_vars_sources.index(idx))

    def _calculate_single_link(
        self,
        data,
        current_value,
        source_vars,
        target_vars=None,
        sources="all",
        conditioning="full",
    ):
        """Calculate dependency measure for all links into a target.

        Calculate dependency measure for all links into a target. A single link
        may consist of information that multiple past variables in a source
        have about the target. The measure can be transfer entropy or mutual
        information and is estimated as the joint information all selected past
        variables from a single source have about the target.

        The conditioning defines which variables are included in the
        conditioning set when estimating a dependency measure. This can be set
        to

        - 'full' to include all selected variables (for multivariate TE this
          includes the target's past variables and past variables from all
          other inferred sources, for multivariate MI this includes past
          variables from all other inferred sources) from all other inferred
          sources and the target's past,
        - 'target' to include variables from the target's past alone (for
          bivariate TE estimation),
        - 'none' for no conditioning (for bivariate MI estimation).

        For transfer entropy, the information transfer is calculated
        conditional on the target's past. For multivariate TE or MI, the
        information (transfer) is calculated conditionally on selected
        variables from further sources in the network.

        Measures can be estimated either for 'all' sources (determined from the
        selected source variables) or for individual sources. A list of
        estimated values for each link (source-target combination) is returned.

        Args:
            data : Data instance
                raw data for analysis
            current_value : tuple
                index of the current value used for estimation, (idx process,
                idx sample)
            source_vars : np array of tuples
                array of past source variables, where one tuple describes a
                single variable as (idx process, idx sample)
            target_vars : np array of tuples [optional]
                array of past target variables
            sources : list of ints | 'all' [optional]
                return estimates for selected sources or all sources (default)
            conditioning : str [optional]
                set conditioning set, 'full' for all selected variables
                (target's and sources' past), 'target' for variables from the
                target's past only, 'none' for no conditioning

        Returns:
            numpy array
                estimate of dependency measure for each link

        Raises:
            ex.AlgorithmExhaustedError
                Raised from estimate() when calculation cannot be made
        """
        # Get realisations of target variables and the current value, constant
        # over sources. Permute current value realisations to generate
        # surrogates if requested.
        current_value_realisations = data.get_realisations(
            current_value, [current_value])

        # Check requested sources.
        if sources == "all":
            sources = np.unique([s[0] for s in source_vars])
        else:
            if type(sources) is int:  # handle integer inputs
                sources = [sources]
            sources = np.array(sources)
            if any(sources > (data.n_processes - 1)):
                raise RuntimeError(
                    "At least one source ({0}) is not in no. "
                    "nodes in the data ({1}).".format(sources, data.n_processes)
                )

        # Allocate memory: either a multidimensional array if local values are
        # required, or a 1D-array for averaged values for each link.
        if self.settings["local_values"]:
            # Collect local values in a [sources x samples x replications]
            # matrix.
            links = np.zeros(
                (
                    len(sources),
                    data.n_realisations_samples(current_value),
                    data.n_replications,
                )
            )
        else:
            links = np.zeros(len(sources))

        # Loop over individual sources.
        for i, s in enumerate(sources):
            # Separate source variables in variables belonging to the current
            # link and variables belonging to the conditioning set. Get
            # realisations for the current link's selected source variables.
            link_vars = [i for i in source_vars if i[0] == s]
            conditional_vars = [i for i in source_vars if i[0] != s]
            source_realisations = data.get_realisations(
                current_value, link_vars)

            # Determine which type of conditioning is requested.
            if conditioning == 'full':
                vars = (conditional_vars if conditional_vars else []) + \
                          (target_vars if target_vars else [])
                conditional_realisations = data.get_realisations(current_value, vars)

            elif conditioning == 'target':  # use target's past only (biv. TE)
                conditional_realisations = data.get_realisations(current_value, target_vars)
            elif conditioning == 'none':  # no conditioning (bivariate MI)
                conditional_realisations = None
            else:
                raise RuntimeError("Unknown conditioning: {0}.".format(conditioning))

            if self.settings["local_values"]:
                local_values = self._cmi_estimator_local.estimate(
                    var1=current_value_realisations,
                    var2=source_realisations,
                    conditional=conditional_realisations,
                )
                links[i] = local_values.reshape(
                    data.n_replications, -1).T
            else:
                links[i] = self._cmi_estimator.estimate(
                    var1=current_value_realisations,
                    var2=source_realisations,
                    conditional=conditional_realisations,
                )

        return links

    def _set_checkpointing_defaults(self, settings, data, sources, target):
        """Set defaults for writing analysis checkpoints."""
        settings.setdefault("write_ckp", False)
        if settings["write_ckp"]:
            settings.setdefault("filename_ckp", "./idtxl_checkpoint")
            filename_ckp = "{0}.ckp".format(settings["filename_ckp"])
            if not os.path.isfile(filename_ckp):
                self._initialise_checkpoint(settings, data, sources, target)
            return settings
        else:
            return settings

    def _initialise_checkpoint(self, settings, data, sources, targets):
        """Write first checkpoint file, data, and settings to disk.

        Called once at the beggining of an analysis using checkpointing. Write
        data and analysis settings to disk. This needs to be done only once.
        Initialise checkpoint file: write header with time stamp, path to data
        and settings, and targets and sources to be analysed. The checkpoint
        file is updated during the analyis.
        """
        # Check if targets is an int, convert to array.
        if type(targets) is int:
            targets = [targets]
        # Write data to disk.
        io.save_pickle(data, "{0}.dat".format(settings["filename_ckp"]))
        # Write settings to disk.
        io.save_json(settings, "{0}.json".format(settings["filename_ckp"]))

        # Initialise checkpoint file for later updates.
        filename_ckp = "{0}.ckp".format(settings["filename_ckp"])
        with open(filename_ckp, "w") as text_file:
            text_file.write("IDTxl checkpoint file.\n")
            timestamp = datetime.now()
            text_file.write("{:%Y-%m-%d %H:%M:%S}\n".format(timestamp))
            text_file.write(
                "Raw data path: {}.dat\n".format(
                    os.path.abspath(settings["filename_ckp"])
                )
            )
            text_file.write(
                "Settings path: {}.json\n".format(
                    os.path.abspath(settings["filename_ckp"])
                )
            )
            text_file.write("Targets to be analyzed: {}\n".format(targets))
            text_file.write("Sources to be analyzed: {}\n\n".format(sources))
            text_file.write(
                "Selected variables (target: [sources]: [selected variables]):"
                "\n{}".format(targets[0])
            )

    def _write_checkpoint(self):
        """Write checkpoint to disk.

        Write checkpoint to disk. The checkpoint contains variables already
        selected by network analysis algorithms. To recover from a checkpoint
        use the 'recover_checkpoint()‘ method.

        Note: IDTxl will always keep the current (*.ckp) and the previous
        version (*.ckp.old) of the checkpoint file to ensure a recoverable
        state even if writing of the current checkpoint fails.
        """
        filename_ckp = "{0}.ckp".format(self.settings["filename_ckp"])

        # Check if a checkpoint file already exists. If yes,
        #   1. make a copy using the same file name plus the .old extension
        #      (overwriting the last *.ckp.old file);
        #   2. update current checkpoint file.
        if os.path.isfile(filename_ckp):
            copyfile(filename_ckp, "{}.old".format(filename_ckp))
            self._update_checkpoint(filename_ckp)
        else:
            raise RuntimeError(
                "Could not find checkpoint file for updating. "
                "Initialise checkpoint first."
            )

    def _update_checkpoint(self, filename_ckp):
        """Update existing checkpoint file.

        Add the last selected variable to the *.ckp file while keeping the
        path to data and settings. Overwrite time stamp in header.
        """
        # We don't expect these files to become very big. Hence, it is the
        # easiest to load the whole file into a data structure and then write
        # it back (https://stackoverflow.com/a/328007). Alternatively, we can
        # just add the last selected variable as a tuple -> then we have to
        # make sure, the last selected candidate always ends up at the end of
        # the selected candidates list.

        # Write time stamp and info
        timestamp = datetime.now()
        # Convert absolute indices to lags with respect to the current value.
        selected_variables = self._idx_to_lag(
            self.selected_vars_full, self.current_value[1]
        )
        # Read file as list of lines and replace first and last line. Write
        # modified file back to disk.
        with open(filename_ckp, "r") as f:
            lines = f.readlines()
        lines[1] = "{:%Y-%m-%d %H:%M:%S}\n".format(timestamp)
        if int(lines[-1][0]) == self.target:
            lines[-1] = "{0}: {1}: {2}\n".format(
                self.target, self.source_set, selected_variables
            )
        else:
            lines.append(
                "{0}: {1}: {2}\n".format(
                    self.target, self.source_set, selected_variables
                )
            )
        with open(filename_ckp, "w") as f:
            f.writelines(lines)

    def resume_checkpoint(self, file_path):
        """Resume analysis from a checkpoint saved to disk.

        Args:
            file_path : str
                path to checkpoint file (excluding extension: .ckp)
        """

        # Read checkpoint
        with open("{}.ckp".format(file_path), "r") as f:
            lines = f.readlines()
        timestamp = lines[1]
        data_path = lines[2][15:].strip()
        settings_path = lines[3][15:].strip()
        # Load settings and data
        data = io.load_pickle(data_path)
        settings = io.load_json(settings_path)
        verbose = settings.get("verbose", True)
        if verbose:
            print(
                "Resuming analysis from file {}.ckp, saved {}".format(
                    file_path, timestamp
                )
            )
        # Read targets and sources.
        targets = ast.literal_eval(lines[4].split(":")[1].strip())
        sources = ast.literal_eval(lines[5].split(":")[1].strip())
        # Read selected variables
        # Format: target - sources analyzed - selected variables
        selected_variables = {}  # vars as lags wrt. the current value
        for l in range(8, len(lines)):
            result = [x.strip() for x in lines[l].split(":")]
            # ast.literal_eval(result[2]): IndexError: list index out of range
            try:
                selected_variables[int(result[0])] = ast.literal_eval(result[2])
            except IndexError:
                if verbose:
                    print("No variables previously selected.")

        if verbose:
            print("Selected variables per target:")
            pprint(selected_variables)

        # Add already selected candidates as conditionals to be added to the
        # settings dict. Note that the time stamp in the selected variables
        # list is a lag wrt. the current value. This format is also expected by
        # the method that manually adds conditionals.
        settings["add_conditionals"] = selected_variables

        return data, settings, targets, sources
