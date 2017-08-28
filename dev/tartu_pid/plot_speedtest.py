"""Plot results of speed test of the PID estimator."""
import numpy as np
import matplotlib.pyplot as plt

datapath = '/home/patriciaw/repos/IDTxl/dev/tartu_pid/speedtest_results/'
temp = np.genfromtxt(datapath + 'and_ci_sydney_run_1.csv')
n_runs = 20
n_data_size = temp.shape[0]


for fun in ['and', 'xor']:
    ci_tartu = np.empty((n_data_size, n_runs))
    ci_sydney = np.empty((n_data_size, n_runs))
    si_tartu = np.empty((n_data_size, n_runs))
    si_sydney = np.empty((n_data_size, n_runs))
    t_tartu = np.empty((n_data_size, n_runs))
    t_sydney = np.empty((n_data_size, n_runs))
    for r in range(n_runs):
        temp = np.genfromtxt(datapath + fun + '_ci_sydney_run_' + str(r + 1) + '.csv')
        ci_sydney[:, r] = temp
        temp = np.genfromtxt(datapath + fun + '_ci_tartu_run_' + str(r + 1) + '.csv')
        ci_tartu[:, r] = temp
        temp = np.genfromtxt(datapath + fun + '_si_sydney_run_' + str(r + 1) + '.csv')
        si_sydney[:, r] = temp
        temp = np.genfromtxt(datapath + fun + '_si_tartu_run_' + str(r + 1) + '.csv')
        si_tartu[:, r] = temp
        temp = np.genfromtxt(datapath + fun + '_t_sydney_run_' + str(r + 1) + '.csv')
        t_sydney[:, r] = temp
        temp = np.genfromtxt(datapath + fun + '_t_tartu_run_' + str(r + 1) + '.csv')
        t_tartu[:, r] = temp

num_samples = [1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]
# num_samples = [100, 200, 500]


# plot error
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(num_samples, (ci_tartu - ci_sydney).mean(axis=1), 'r')
plt.plot(num_samples, (si_tartu - si_sydney).mean(axis=1), 'b')
plt.show()
plt.xlabel('N')
plt.ylabel('Error: tartu - sydney')

# plot error
plt.subplot(122)
h1, = plt.plot(num_samples, t_sydney.mean(axis=1), 'r')
h2, = plt.plot(num_samples, t_tartu.mean(axis=1), 'b')
plt.show()
plt.xlabel('N')
plt.ylabel('time [s]')
plt.legend([h1, h2], ['Sydney', 'Tartu'])
