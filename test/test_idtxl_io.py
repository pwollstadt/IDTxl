"""Unit tests for IDTxl I/O functions."""
import os
from idtxl import idtxl_io as io
from idtxl.data import Data
from idtxl.single_process_storage import Single_process_storage


def test_save_te_results():
    """Test saving of TE results."""
    # Generate some example output
    dat = Data()
    dat.generate_mute_data(100, 2)
    max_lag = 5
    analysis_opts = {
        'cmi_calc_name': 'opencl_kraskov',
        'n_perm_mi': 22,
        'alpha_mi': 0.05,
        'tail_mi': 'one',
        }
    processes = [2, 3]
    network_analysis = Single_process_storage(max_lag, analysis_opts, tau=1)
    res_ais = network_analysis.analyse_network(dat, processes)

    cwd = os.getcwd()
    fp = ''.join([cwd, '/idtxl_unit_test/'])
    if not os.path.exists(fp):
        os.makedirs(fp)
    io.save(res_ais, file_path=''.join([fp, 'res_ais']))
    f = io.load(file_path=''.join([fp, 'res_single.txt']))
    print('THIS MODULE IS NOT YET WORKING!')
    # assert (f is not None), 'File read from disk is None.'


if __name__ == '__main__':
    # test_save_te_results()
