#!/bin/bash
#
# performancetest_lorenz_2.py with several different settings on a SLURM batch system.
# Submit using command "sbatch".
#
#SBATCH -A cidbn
#SBATCH -p cidbn
#SBATCH --job-name=te_mp
#SBATCH --output=performancetest_lorenz_2_javamultithread_%A.txt
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128gb
#SBATCH --exclusive

cd /usr/users/$USER/IDTxl

export PYTHONPATH=/usr/users/$USER/IDTxl
export JAVA_HOME=/usr/users/$USER/jdk-16.0.1

source ~/.bashrc
conda activate idtxl

date

# Run tests with a single java thread
for n_java_threads in 64 32 16 8 4 2 1
do
    python test/performancetest_lorenz_2.py --n_java_threads=$n_java_threads
    wait
    date
done
