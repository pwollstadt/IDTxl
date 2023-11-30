"""Provides unit test for data_spiketimes class."""

from idtxl.data_spiketime import Data_spiketime
import numpy as np
import os



def test_set_data():
    """Test if data is written correctly into a Data instance."""
    # add one process to data
    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                         'data/spike_times.dat'), dtype=float)
    spiketimedata = np.empty(shape=(1), dtype=np.ndarray)
    spiketimedata[0] = spiketimes

    data1 = Data_spiketime()
    data1.set_data(spiketimedata)

    assert(data1.data[0].T == spiketimes.T-min(spiketimes)).all(), ('Class data does not match '
                                                                    'input spikketimes.')
    # add multiple processes to data
    nr_processes = 10
    spiketimedata2 = np.empty(shape=(nr_processes), dtype=np.ndarray)

    for i in range(nr_processes):
        if i == 0:
            spiketimedata2[i] = spiketimes
        else:
            ran = np.random.rand(len(spiketimes)) * 1000
            new = spiketimes + ran
            sampl = int(np.random.uniform(low=0.6 * len(spiketimes), high=0.9 * len(spiketimes), size=(1,)))
            spiketimedata2[i] = new[0:sampl]

    data2 = Data_spiketime()
    data2.set_data(spiketimedata2)

    for i in range(nr_processes):
        assert(data2.data[i].T == spiketimedata2[i].T-min(spiketimedata2[i])).all(), ('Class data does not match '
                                                                                       'input spikketimes.')


def test_load_Rudelt():
    """Test loading Rudelt data."""
    data = Data_spiketime()  # initialise empty data object
    data.load_Rudelt_data()  # load Rudelt spike time data


    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                     'data/spike_times.dat'), dtype=float)


    assert (data.data[0].T == spiketimes.T - min(spiketimes)).all(), ('Class data does not match '
                                                                        'input spikketimes.')


def test_get_realisations_symbols():
    """Test low-level function for data retrieval."""
    # add multiple processes to data
    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                         'data/spike_times.dat'), dtype=float)
    nr_processes = 10
    spiketimedata2 = np.empty(shape=(nr_processes), dtype=np.ndarray)
    nr_spikes = np.empty(shape=(nr_processes), dtype=int)

    for i in range(nr_processes):
        if i == 5:
            spiketimedata2[i] = spiketimes
            nr_spikes[i] = len(spiketimes)
        else:
            ran = np.random.rand(len(spiketimes)) * 1000
            new = spiketimes + ran
            sampl = int(np.random.uniform(low=0.6 * len(spiketimes), high=0.9 * len(spiketimes), size=(1,)))
            nr_spikes[i] = sampl
            spiketimedata2[i] = new[0:sampl]

    data = Data_spiketime()
    data.set_data(spiketimedata2)

    # get realisation of single process
    process_list = 5
    past_range_T = 0.31548
    number_of_bins_d = 5
    scaling_k = 0.0
    embedding_step_size = 0.05

    symbol_array, past_symbol_array, current_symbol_array, symbol_array_length, orig_spiketimes = \
        data.get_realisations_symbols(process_list,
                                      past_range_T,
                                      number_of_bins_d,
                                      scaling_k,
                                      embedding_step_size,
                                      output_spike_times=True)

    assert (orig_spiketimes[0].T == spiketimes.T - min(spiketimes)).all(), ('Class ouput data does not match '
                                                                            'input spiketimes.')
    assert (len(symbol_array) == 1), ('symbol array length does not match with '
                                      'number of processes given')
    assert (len(past_symbol_array) == 1), ('past symbol array length does not match with '
                                           'number of processes given')
    assert (len(current_symbol_array) == 1), ('current symbol array length does not match with '
                                              'number of processes given')

    assert (len(symbol_array[0]) == symbol_array_length[0]), \
        ('length of symbol array does not match with '
            'with the extracted number of symbols')
    assert (len(symbol_array[0]) == len(past_symbol_array[0])), \
        ('length of symbol array does not match with '
                                                                 'with length of past symbol array')
    assert (len(symbol_array[0]) == len(current_symbol_array[0])), ('length of symbol array does not match with '
                                                                    'with length of current symbol array')

    # get realisations of multiple processes
    process_list = [1, 3, 5, 7, 8]

    symbol_array2, past_symbol_array2, current_symbol_array2, symbol_array_length2, orig_spiketimes2 = \
        data.get_realisations_symbols(process_list,
                                      past_range_T,
                                      number_of_bins_d,
                                      scaling_k,
                                      embedding_step_size,
                                      output_spike_times=True)

    assert (len(symbol_array2) == len(process_list)), \
        ('symbol array length does not match with '
            'number of processes given')
    assert (len(past_symbol_array2) == len(process_list)), \
        ('past symbol array length does not match with '
            'number of processes given')
    assert (len(current_symbol_array2) == len(process_list)), \
        ('current symbol array length does not match with '
            'number of processes given')
    for ii in range(len(process_list)):
        assert (len(symbol_array2[ii]) == symbol_array_length2[ii]), \
            ('length of symbol array does not match with '
                'with the extracted number of symbols')
        assert (len(symbol_array2[ii]) == len(past_symbol_array2[ii])), \
            ('length of symbol array does not match with '
                'with length of past symbol array')
        assert (len(symbol_array2[ii]) == len(current_symbol_array2[ii])), \
            ('length of symbol array does not match with '
                'with length of current symbol array')


def test_get_bootstrap_realisations():
    """Test low-level function for data retrieval."""
    # add multiple processes to data
    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                         'data/spike_times.dat'), dtype=float)
    nr_processes = 10
    spiketimedata2 = np.empty(shape=(nr_processes), dtype=np.ndarray)
    nr_spikes = np.empty(shape=(nr_processes), dtype=int)

    for i in range(nr_processes):
        if i == 5:
            spiketimedata2[i] = spiketimes
            nr_spikes[i] = len(spiketimes)
        else:
            ran = np.random.rand(len(spiketimes)) * 1000
            new = spiketimes + ran
            sampl = int(np.random.uniform(low=0.6 * len(spiketimes), high=0.9 * len(spiketimes), size=(1,)))
            nr_spikes[i] = sampl
            spiketimedata2[i] = new[0:sampl]

    data = Data_spiketime()
    data.set_data(spiketimedata2)

    # get bootstrap realisation of single process
    process_list = 5
    past_range_T = 0.31548
    number_of_bins_d = 5
    scaling_k = 0.0
    embedding_step_size = 0.05

    bs_symbol_array, bs_past_symbol_array, bs_current_symbol_array = \
        data.get_bootstrap_realisations_symbols(process_list,
                                      past_range_T,
                                      number_of_bins_d,
                                      scaling_k,
                                      embedding_step_size)

    assert (len(bs_symbol_array) == 1), \
        ('symbol array length does not match with '
            'number of processes given')
    assert (len(bs_past_symbol_array) == 1), \
        ('past symbol array length does not match with '
            'number of processes given')
    assert (len(bs_current_symbol_array) == 1), \
        ('current symbol array length does not match with '
            'number of processes given')

    assert (len(bs_symbol_array[0]) == len(bs_past_symbol_array[0])), \
        ('length of symbol array does not match with '
            'with length of past symbol array')
    assert (len(bs_symbol_array[0]) == len(bs_current_symbol_array[0])), \
        ('length of symbol array does not match with '
            'with length of current symbol array')

    # test manual block length input
    bs_symbol_array, bs_past_symbol_array, bs_current_symbol_array = \
        data.get_bootstrap_realisations_symbols(process_list,
                                                past_range_T,
                                                number_of_bins_d,
                                                scaling_k,
                                                embedding_step_size,
                                                symbol_block_length=49)
    bs_symbol_array, bs_past_symbol_array, bs_current_symbol_array = \
        data.get_bootstrap_realisations_symbols(process_list,
                                                past_range_T,
                                                number_of_bins_d,
                                                scaling_k,
                                                embedding_step_size,
                                                symbol_block_length=30)

    # get bootstrap realisations of multiple processes
    process_list = [1, 7, 5, 3, 9]

    bs_symbol_array2, bs_past_symbol_array2, bs_current_symbol_array2 = \
        data.get_bootstrap_realisations_symbols(process_list,
                                                past_range_T,
                                                number_of_bins_d,
                                                scaling_k,
                                                embedding_step_size)

    assert (len(bs_symbol_array2) == len(process_list)), \
        ('symbol array length does not match with '
            'number of processes given')
    assert (len(bs_past_symbol_array2) == len(process_list)), \
        ('past symbol array length does not match with '
            'number of processes given')
    assert (len(bs_current_symbol_array2) == len(process_list)), \
        ('current symbol array length does not match with '
            'number of processes given')
    for ii in range(len(process_list)):
        assert (len(bs_symbol_array2[ii]) == len(bs_past_symbol_array2[ii])), \
            ('length of symbol array does not match with '
                'with length of past symbol array')
        assert (len(bs_symbol_array2[ii]) == len(bs_current_symbol_array2[ii])), \
            ('length of symbol array does not match with '
                'with length of current symbol array')


def test_get_recording_length():
    data = Data_spiketime()  # initialise empty data object
    data.load_Rudelt_data()  # load Rudelt spike time data

    record_length = data.get_recording_length(0)

    assert (abs(record_length - 952.2185) == 0), \
        'Record length does not match the example data'


def test_get_firingrate():
    data = Data_spiketime()  # initialise empty data object
    data.load_Rudelt_data()  # load Rudelt spike time data

    firingrate = data.get_firingrate(0, 0.005)

    assert (abs(firingrate - 4.020079393417488) == 0), \
        'Firing rate does not match the example data'


def test_get_H_spiking():
    data = Data_spiketime()  # initialise empty data object
    data.load_Rudelt_data()  # load Rudelt spike time data

    Hspiking = data.get_H_spiking(0, 0.005)
    assert (abs(Hspiking - 0.09842958352312628) == 0), ('Entropy does not match the one of the example data')


def test_get_realisations():
    """Test low-level function for data retrieval."""
    # add multiple processes to data
    spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                         'data/spike_times.dat'), dtype=float)
    nr_processes = 10
    spiketimedata2 = np.empty(shape=(nr_processes), dtype=np.ndarray)
    nr_spikes = np.empty(shape=(nr_processes), dtype=int)

    for i in range(nr_processes):
        if i == 5:
            spiketimedata2[i] = spiketimes
            nr_spikes[i] = len(spiketimes)
        else:
            ran = np.random.rand(len(spiketimes)) * 1000
            new = spiketimes + ran
            sampl = int(np.random.uniform(low=0.6 * len(spiketimes), high=0.9 * len(spiketimes), size=(1,)))
            nr_spikes[i] = sampl
            spiketimedata2[i] = new[0:sampl]

    data = Data_spiketime()
    data.set_data(spiketimedata2)

    # get realisation of single process
    process_list = 5

    orig_spiketimes = \
        data.get_realisations(process_list)

    assert (orig_spiketimes[0].T == spiketimes.T - min(spiketimes)).all(), ('Class ouput data does not match '
                                                                            'input spiketimes.')

    # get realisations of multiple processes
    process_list = [1, 3, 5, 7, 8]

    orig_spiketimes2 = data.get_realisations(process_list)

    assert (len(orig_spiketimes2) == len(process_list)), \
        ('length of spike time array does not match with '
         'number of processes given')
    for i in range(len(process_list)):
        assert (orig_spiketimes2[i].T == spiketimedata2[process_list[i]].T - min(spiketimedata2[process_list[i]])).all(), \
                'extracted spike times do not match the input spike times'


if __name__ == '__main__':
    test_set_data()
    test_load_Rudelt()
    test_get_realisations_symbols()
    test_get_bootstrap_realisations()
    test_get_firingrate()
    test_get_H_spiking()
    test_get_realisations()
