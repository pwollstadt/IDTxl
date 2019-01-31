"""Provide unit tests for IDTxl checkpointing."""
import os
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data


def test_checkpoint_resume():
    """Test resuming from manually generated checkpoint."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)

    # Initialise analysis object and define settings
    filename_ckp = './my_checkpoint'
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Manually create checkpoint files for two targets.
    # Note that variables are added as absolute time indices and not lags wrt.
    # the current value. Internally, IDTxl operates with abolute indices, but in
    # all results and console outputs, variables are described by their lags
    # wrt. to the current value.
    sources = [0, 1, 2]
    targets = [3, 4]
    add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    network_analysis._set_checkpointing_defaults(
        settings, data, sources, targets)
    for i in range(len(targets)):
        network_analysis._initialise(settings, data, sources, targets[i])
        network_analysis.selected_vars_full = add_vars[i]
        network_analysis._update_checkpoint('{}.ckp'.format(filename_ckp))

    # Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp)

    # Test if variables were added correctly
    for i in range(len(targets)):
        network_analysis_res._initialise(settings, data, sources, targets[i])
        assert network_analysis_res.selected_vars_full == add_vars[i], (
            'Variables were not added correctly for target {0}.'.format(
                targets[i]))

    # Test if analysis runs from resumed checkpoint.
    results = network_analysis_res.analyse_network(
        settings=settings, data=data, targets=targets, sources=sources)

    # Remove test files.
    os.remove('{0}.ckp'.format(filename_ckp))
    os.remove('{0}.ckp.old'.format(filename_ckp))
    os.remove('{0}.dat'.format(filename_ckp))
    os.remove('{0}.json'.format(filename_ckp))

def test_checkpoint_copy():
    """Assert that a copy of the current checkpoint is written upon update."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)

    # Initialise analysis object and define settings
    filename_ckp = './my_checkpoint'
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Manually create checkpoint files for two targets.
    # Note that variables are added as absolute time indices and not lags wrt.
    # the current value. Internally, IDTxl operates with abolute indices, but in
    # all results and console outputs, variables are described by their lags
    # wrt. to the current value.
    sources = [0, 1, 2]
    targets = [3, 4]
    add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    network_analysis._initialise(settings, data, sources, targets[0])
    network_analysis._write_checkpoint()

    assert os.path.isfile('{0}.ckp'.format(filename_ckp)), (
        'No checkpoint file {}.ckp was written.'.format(filename_ckp))
    assert os.path.isfile('{0}.ckp.old'.format(filename_ckp)), (
        'No checkpoint file {}.ckp.old was written.'.format(filename_ckp))

    # Remove test files.
    os.remove('{0}.ckp'.format(filename_ckp))
    os.remove('{0}.ckp.old'.format(filename_ckp))
    os.remove('{0}.dat'.format(filename_ckp))
    os.remove('{0}.json'.format(filename_ckp))


if __name__ == '__main__':
    test_checkpoint_copy()
    test_checkpoint_resume()
