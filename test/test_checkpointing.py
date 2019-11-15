"""Provide unit tests for IDTxl checkpointing."""
import os
import subprocess
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.network_comparison import NetworkComparison
import copy
import time


def test_checkpoint_resume():
    """Test resuming from manually generated checkpoint."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)


    # Initialise analysis object and define settings
    filename_ckp = './my_checkpoint'
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtKraskovCMI',
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
        print(i)
        network_analysis.selected_vars_full = add_vars[i]
        network_analysis._update_checkpoint('{}.ckp'.format(filename_ckp))

    # Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp)

    # Test if variables were added correctly
    #for i in range(len(targets)):
    #    network_analysis_res._initialise(settings, data, sources, targets[i])
    #    assert network_analysis_res.selected_vars_full == add_vars[i], (
    #        'Variables were not added correctly for target {0}.'.format(
    #            targets[i]))

    # Test if analysis runs from resumed checkpoint.
    results = network_analysis_res.analyse_network(
        settings=settings, data=data, targets=targets, sources=sources)

    # Remove test files.
    #os.remove('{0}.ckp'.format(filename_ckp))
    #os.remove('{0}.ckp.old'.format(filename_ckp))
    #os.remove('{0}.dat'.format(filename_ckp))
    #os.remove('{0}.json'.format(filename_ckp))

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
    #add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    network_analysis._initialise(settings, data, sources, targets)
    network_analysis._write_checkpoint()

    #assert os.path.isfile('{0}.ckp'.format(filename_ckp)), (
    #    'No checkpoint file {}.ckp was written.'.format(filename_ckp))
    #assert os.path.isfile('{0}.ckp.old'.format(filename_ckp)), (
    #    'No checkpoint file {}.ckp.old was written.'.format(filename_ckp))

    # Remove test files.
    #os.remove('{0}.ckp'.format(filename_ckp))
    #os.remove('{0}.ckp.old'.format(filename_ckp))
    #os.remove('{0}.dat'.format(filename_ckp))
    #os.remove('{0}.json'.format(filename_ckp))

def test_checkpoint_equal():
    """Assert that a result from a resumed checkpoint is equal to the original start."""
    filename_ckp = './my_checkpoint'
    filepath = "test/save_all_checkpoints.sh"
    # Running Script to save all Checkpoints
    subprocess.Popen(['./' + filepath, filename_ckp  + ".ckp.old"])

    """Test resuming from manually generated checkpoint."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Setting sources and targets for the analyis
    sources = [0, 1, 2]
    targets = [3, 4]
    #add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    #network_analysis._initialise(settings, data, sources, targets[0])
    #network_analysis._initialise(settings, data, sources, targets[1])

    # Starting Analysis of the Network
    results = network_analysis.analyse_network(
        settings=settings, data=data, targets=targets, sources='all')

    #Closing running script which saves all checkpoints seperately
    os.system("ps -ef | grep save_all_checkpoints.sh | grep -v grep | awk '{print $2}' | xargs kill")
    proc = subprocess.Popen(["ls -dq *my_checkpoint.ckp.old* | wc -l "], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    checkpoints = int(out)

    #Resuming Analysis from every checkpoint created
    #os.system("mv my_checkpoint.ckp cmp_checkpoint.ckp")
    for i in range(2, checkpoints + 1):
        check = './my_checkpoint.ckp.old'

        # cannot resume first checkpoint properly
        print(i)
        #print(check+str(1))
        data, settings, targets, sources = network_analysis.resume_checkpoint(check+str(i))

    #data, settings, targets, sources = network_analysis.resume_checkpoint("my_checkpoint.ckp.old14")
        results = network_analysis.analyse_network(
            settings=settings, data=data, targets=targets, sources='all')
        os.system("mv my_checkpoint.ckp cmp_checkpoint.ckp" + str(i))

     #first checkpoint does not include any sources

def testing():
     # Generate test data
     data = Data()
     data.generate_mute_data(n_samples=10000, n_replications=5)

     # Initialise analysis object and define settings
     network_analysis = MultivariateTE()
     network_analysis2 = MultivariateTE()
     network_analysis3 = MultivariateTE()
     settings = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 5,
                 'min_lag_sources': 1
                 }

     # Setting sources and targets for the analyis
     sources = [0, 1, 2]
     targets = [3, 4]

     # Starting Analysis of the Network
     results = network_analysis.analyse_network(
         settings=settings, data=data, targets=targets, sources=sources)

     results2 = network_analysis2.analyse_network(
         settings=settings, data=data, targets=targets, sources=sources)
     print(network_analysis == network_analysis)
     print(network_analysis == network_analysis2)

     print("====================================== results2 ======================================================")
     settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 5,
                 'min_lag_sources': 1
                 }
     data2 = Data()
     data2.generate_mute_data(n_samples=1000, n_replications=5)
     sources2 = [0, 1]
     targets2 = [2, 3, 4]
     results3 = network_analysis3.analyse_network(
           settings=settings2, data=data2, targets=targets2, sources=sources2)

     comp_settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'stats_type': 'independent',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 50,
        'n_perm_max_seq': 50,
        'alpha_comp': 0.26,
        'n_perm_comp': 200,
        'tail': 'two',
        'permute_in_time': True,
        'perm_type': 'random'
    }

     #comp = NetworkComparison()
     #results42 = comp.compare_within(comp_settings, results, results3, data, data2)
     #adj_matrix = results42.get_adjacency_matrix(weights='comparison')
     #adj_matrix.print_matrix()
     adj_matrix1 = results.get_adjacency_matrix(weights='max_te_lag', fdr=False)
     adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
     adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
     print(adj_matrix1 == adj_matrix2)
     print(adj_matrix1 == adj_matrix3)
     print(adj_matrix2 == adj_matrix3)

     adj_matrix1.print_matrix()
     adj_matrix2.print_matrix()
     adj_matrix3.print_matrix()
     print(adj_matrix1.print_matrix() == adj_matrix2.print_matrix())
     print(adj_matrix2.print_matrix() == adj_matrix3.print_matrix())

def cmp_test():

    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=10000, n_replications=5)

    data2 = copy.deepcopy(data)
    data3 = copy.deepcopy(data)
    data4 = copy.deepcopy(data)
    data5 = copy.deepcopy(data)

    # Initialise analysis object and define settings
    network_analysis1 = MultivariateTE()
    network_analysis2 = MultivariateTE()
    network_analysis3 = MultivariateTE()
    network_analysis4 = MultivariateTE()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data2, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data3, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = MultivariateTE()
    data3, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data3, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data4, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data5, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    #os.remove('{0}.ckp'.format(filename_ckp1))
    #os.remove('{0}.ckp.old'.format(filename_ckp1))
    #os.remove('{0}.dat'.format(filename_ckp1))
    #os.remove('{0}.json'.format(filename_ckp1))

    #os.remove('{0}.ckp'.format(filename_ckp2))
    #os.remove('{0}.ckp.old'.format(filename_ckp2))
    #os.remove('{0}.dat'.format(filename_ckp2))
    #os.remove('{0}.json'.format(filename_ckp2))


if __name__ == '__main__':
    #test_checkpoint_copy()
    #test_checkpoint_resume()
    #test_checkpoint_equal()
    #testing()
    cmp_test()



"""
example 1:
printing results
result 1: without checkpoint
[(0, 2, 3) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 2: with checkpoint
[(0, 2, 3) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 3: resuming from checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 4: without checkpoint with different targets and sources
[(0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 5: without checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
printing if results are equal
comparing result 1 and 2
[ True  True  True  True]
comparing result 1 and 3
test/test_checkpointing.py:329: DeprecationWarning: elementwise == comparison failed; this will raise an error in the future.
  print(result1 == result3)
False
comparing result 1 and 4
[False  True  True  True]
comparing result 1 and 5
test/test_checkpointing.py:333: DeprecationWarning: elementwise == comparison failed; this will raise an error in the future.
  print(result1 == result5)
False
printing length
length of result1:
4
length of result2:
4
length of result3:
5
length of result4:
4
length of result5:
5
printing comparison of length
length of comparison 1-2 to 1
True
length of comparison 1-3 to 1
False
length of comparison 1-4 to 1
False
length of comparison 1-5 to 1
False
final
elements for comparison between result 1 and 2
[(0, 2, 3), (1, 3, 1), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 3
[(0, 2, 3)]
elements for comparison between result 1 and 4
[(1, 3, 1), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 5
[(0, 2, 3)]

example 2:
printing results
result 1: without checkpoint
[(0, 2, 3) (0, 3, 1) (1, 3, 3) (3, 4, 1) (4, 3, 1)]
result 2: with checkpoint
[(0, 2, 3) (0, 3, 1) (1, 3, 3) (3, 4, 1) (4, 3, 1)]
result 3: resuming from checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 4: without checkpoint with different targets and sources
[(0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 5: without checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
printing if results are equal
comparing result 1 and 2
[ True  True  True  True  True]
comparing result 1 and 3
[ True False False  True  True]
comparing result 1 and 4
test/test_checkpointing.py:331: DeprecationWarning: elementwise == comparison failed; this will raise an error in the future.
  print(result1 == result4)
False
comparing result 1 and 5
[ True False False  True  True]
printing length
length of result1:
5
length of result2:
5
length of result3:
5
length of result4:
4
length of result5:
5
printing comparison of length
length of comparison 1-2 to 1
True
length of comparison 1-3 to 1
False
length of comparison 1-4 to 1
False
length of comparison 1-5 to 1
False
final
elements for comparison between result 1 and 2
[(0, 2, 3), (0, 3, 1), (1, 3, 3), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 3
[(0, 2, 3), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 4
[]
elements for comparison between result 1 and 5
[(0, 2, 3), (3, 4, 1), (4, 3, 1)]

example 3:
printing results
result 1: without checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 2: with checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 3: resuming from checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 4: without checkpoint with different targets and sources
[(0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 5: without checkpoint
[(0, 2, 3) (0, 3, 2) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
printing if results are equal
comparing result 1 and 2
[ True  True  True  True  True]
comparing result 1 and 3
[ True  True  True  True  True]
comparing result 1 and 4
test/test_checkpointing.py:331: DeprecationWarning: elementwise == comparison failed; this will raise an error in the future.
  print(result1 == result4)
False
comparing result 1 and 5
[ True  True  True  True  True]
printing length
length of result1:
5
length of result2:
5
length of result3:
5
length of result4:
4
length of result5:
5
printing comparison of length
length of comparison 1-2 to 1
True
length of comparison 1-3 to 1
True
length of comparison 1-4 to 1
False
length of comparison 1-5 to 1
True
final
elements for comparison between result 1 and 2
[(0, 2, 3), (0, 3, 2), (1, 3, 1), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 3
[(0, 2, 3), (0, 3, 2), (1, 3, 1), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 4
[]
elements for comparison between result 1 and 5
[(0, 2, 3), (0, 3, 2), (1, 3, 1), (3, 4, 1), (4, 3, 1)]


example 4 (sample length 10000 and deepcopies):

result 1: without checkpoint
[(0, 2, 3) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 2: with checkpoint
[(0, 2, 3) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 3: resuming from checkpoint
[(0, 2, 3) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 4: without checkpoint with different targets and sources
[(1, 3, 1) (3, 4, 1) (4, 3, 1)]
result 5: without checkpoint
[(0, 2, 3) (1, 3, 1) (3, 4, 1) (4, 3, 1)]
printing if results are equal
comparing result 1 and 2
[ True  True  True  True]
comparing result 1 and 3
[ True  True  True  True]
comparing result 1 and 4
test/test_checkpointing.py:363: DeprecationWarning: elementwise == comparison failed; this will raise an error in the future.
  print(result1 == result4)
False
comparing result 1 and 5
[ True  True  True  True]
printing length
length of result1:
4
length of result2:
4
length of result3:
4
length of result4:
3
length of result5:
4
printing comparison of length
length of comparison 1-2 to 1
True
length of comparison 1-3 to 1
True
length of comparison 1-4 to 1
False
length of comparison 1-5 to 1
True
final
elements for comparison between result 1 and 2
[(0, 2, 3), (1, 3, 1), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 3
[(0, 2, 3), (1, 3, 1), (3, 4, 1), (4, 3, 1)]
elements for comparison between result 1 and 4
[]
elements for comparison between result 1 and 5
[(0, 2, 3), (1, 3, 1), (3, 4, 1), (4, 3, 1)]

"""
