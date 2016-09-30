"""Unit tests for IDTxl I/O functions."""
import os
from idtxl import idtxl_io as io
from idtxl.data import Data
from idtxl.multivariate_te import Multivariate_te


def test_save_te_results():
    """Test saving of TE results."""
    # Generate some example output
    data = Data()
    data.generate_mute_data(n_replications=2, n_samples=500)
    print('Demo data with {0} procs, {1} samples, {2} reps.'.format(
                    data.n_processes, data.n_samples, data.n_replications))
    opts = {'cmi_calc_name': 'jidt_kraskov'}
    mte = Multivariate_te(max_lag_sources=3, max_lag_target=3,
                          min_lag_sources=1, options=opts)
    res_single = mte.analyse_single_target(data=data, target=3)
    # res_full = mte.analyse_network(data=data, targets=[0, 1])

    cwd = os.getcwd()
    fp = ''.join([cwd, '/idtxl_unit_test/'])
    if not os.path.exists(fp):
        os.makedirs(fp)
    io.save(res_single, file_path=''.join([fp, 'res_single']))
    f = io.load(file_path=''.join([fp, 'res_single.txt']))
    assert (f is not None), 'File read from disk is None.'


if __name__ == '__main__':
    test_save_te_results()
