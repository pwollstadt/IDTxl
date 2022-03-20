# Import classes
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import matplotlib.pyplot as plt
import time
import sys

from mpi4py import MPI

def main(args):
    """Demo of how to use the MPIEstimator for network analysis
    Call this script using
    mpiexec -n 1 python demos/demo_multivariate_te_mpi.py <number of workers>
    for systems supporting MPI Spawn (MPI version >= 2) or
    mpiexec -n <number of workers + 1> python -m mpi4py.futures demo/demo_multivariate_te_mpi.py <number of workers>
    for legacy MPI 1 implementations.
    
    Call
    python demos/demo_multivariate_te_mpi.py 0
    for a comparison without MPI.

    """
    assert MPI.COMM_WORLD.Get_rank() == 0 

    max_workers = int(args[1])
    print(f'Running TE Test with {max_workers} MPI workers.')

    # a) Generate test data
    data = Data(seed=12345)
    data.generate_mute_data(n_samples=1000, n_replications=5)

    # b) Initialise analysis object and define settings
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'MPI': max_workers > 0,
                'max_workers': max_workers
                }

    # c) Run analysis
    start = time.time()
    results = network_analysis.analyse_network(settings=settings, data=data)
    end = time.time()
    print(f'On {max_workers} workers, the task took {end - start} seconds.')

    # d) Plot inferred network to console and via matplotlib
    results.print_edge_list(weights='max_te_lag', fdr=False)
    plot_network(results=results, weights='max_te_lag', fdr=False)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)