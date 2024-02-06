import os
import time
import argparse
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

"""
This script uses the Lorenz 2 example to test the runtime scaling using multiprocessing and mpi.
"""


def profile_lorenz_2(n_java_threads, multiprocessing, n_processes, mpi, n_workers):
    assert (
        multiprocessing or n_processes == 1
    ), "n_processes must be 1 if multithreading is disabled"

    start_time = time.perf_counter()
    # load simulated data from 2 coupled Lorenz systems 1->2, u = 45 ms
    d = np.load(
        os.path.join(os.path.dirname(__file__), "data/lorenz_2_exampledata.npy")
    )
    data = Data()
    data.set_data(d[:, :, 0:10], "psr")
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 50,
        "min_lag_sources": 40,
        "max_lag_target": 30,
        "tau_sources": 1,
        "tau_target": 3,
        "n_perm_max_stat": 200,
        "n_perm_min_stat": 200,
        "n_perm_omnibus": 500,
        "n_perm_max_seq": 500,
        "num_threads": n_java_threads,
        "multiprocessing": multiprocessing,
        "n_processes": n_processes,
        "MPI": mpi,
        "max_workers": n_workers,
    }
    lorenz_analysis = MultivariateTE()
    _ = lorenz_analysis.analyse_single_target(settings, data, 0)
    runtime = time.perf_counter() - start_time

    print(
        f"Lorenz 2 with {n_java_threads} Java threads and {n_processes} Python processes ({multiprocessing=}) took {runtime:.2f} seconds"
    )

    # Create results file if it doesn't exist
    if not os.path.exists("performancetest_lorenz_2_runtimes.csv"):
        with open("performancetest_lorenz_2_runtimes.csv", "w") as f:
            f.write(
                "n_java_threads,multiprocessing,n_processes,mpi,max_workers,runtime\n"
            )

    # Append runtime to file
    with open("performancetest_lorenz_2_runtimes.csv", "a") as f:
        f.write(
            f"{n_java_threads},{multiprocessing},{n_processes},{mpi},{n_workers},{runtime}\n"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_java_threads", type=int, default=1)
    argparser.add_argument("--multiprocessing", type=bool, default=False)
    argparser.add_argument("--n_processes", type=int, default=1)
    argparser.add_argument("--mpi", type=bool, default=False)
    argparser.add_argument("--n_workers", type=int, default=1)

    args = argparser.parse_args()
    print(
        f"Running Lorenz 2 with {args.n_java_threads} Java threads and {args.n_processes} Python processes (multiprocessing={args.multiprocessing}), on {args.n_workers} workers (MPI={args.mpi}))"
    )
    profile_lorenz_2(
        args.n_java_threads,
        args.multiprocessing,
        args.n_processes,
        args.mpi,
        args.n_workers,
    )
