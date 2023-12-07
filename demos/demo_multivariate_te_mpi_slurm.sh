#!/bin/bash
#
# This bash script runs the demo_multivariate_te_mpi.py with several different settings on a SLURM batch system.
# Submit using command "sbatch demo_multivariate_te_mpi_slurm.sh".
#
#SBATCH --job-name=te_mpi
#SBATCH --output=demo_multivariate_te_mpi_res.txt
#SBATCH --time=2:00:00
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb

cd /usr/users/$USER

export PYTHONPATH=/usr/users/$USER/IDTxl
export JAVA_HOME=/usr/users/$USER/jdk-16.0.1

date

# Run on four nodes using MPI
srun --nodes 4 -n 4 --mpi=pmi2 python -m mpi4py.futures "IDTxl/demos/demo_multivariate_te_mpi.py" 3
wait
date

# Run on three nodes using MPI
srun --nodes 3 -n 3 --mpi=pmi2 python -m mpi4py.futures "IDTxl/demos/demo_multivariate_te_mpi.py" 2
wait
date

# Run on two nodes using MPI
srun --nodes 2 -n 2 --mpi=pmi2 python -m mpi4py.futures "IDTxl/demos/demo_multivariate_te_mpi.py" 1
wait
date

# Run on one node not using MPI
srun --nodes 1 -n 1 --mpi=pmi2 python -m mpi4py.futures "IDTxl/demos/demo_multivariate_te_mpi.py" 0
wait
date
