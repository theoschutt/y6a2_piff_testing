#!/bin/bash
#SBATCH --account=des
#SBATCH --qos=regular
#SBATCH --license=cscratch1
#SBATCH --constraint=haswell
#SBATCH --nodes=20
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=15:00:00


cd /global/cscratch1/sd/schutt20/y6a2_piff_testing/scripts/
conda activate mpi_test

srun --cpu-bind=cores python run_tau_stats.py > run_tau_batch_4.log
