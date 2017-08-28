"""Unit tests for IDTxl I/O functions."""
import os
from idtxl import idtxl_io as io
from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage
from test_estimators_jidt import jpype_missing


# @jpype_missing
# def test_save_te_results():
#     """Test saving of TE results."""
#     # Generate some example output
#     dat = Data()
#     dat.generate_mute_data(100, 2)
#     settings = {
#         'cmi_estimator': 'JidtKraskovCMI',
#         'n_perm_mi': 22,
#         'alpha_mi': 0.05,
#         'tail_mi': 'one',
#         }
#     processes = [2, 3]
#     network_analysis = ActiveInformationStorage(max_lag=5,
#                                                 tau=1,
#                                                 options=settings)
#     res_ais = network_analysis.analyse_network(dat, processes)

#     cwd = os.getcwd()
#     fp = ''.join([cwd, '/idtxl_unit_test/'])
#     if not os.path.exists(fp):
#         os.makedirs(fp)
#     io.save(res_ais, file_path=''.join([fp, 'res_ais']))
#     f = io.load(file_path=''.join([fp, 'res_single.txt']))
#     print('THIS MODULE IS NOT YET WORKING!')
#     assert (f is not None), 'File read from disk is None.'


# if __name__ == '__main__':
#     test_save_te_results()
