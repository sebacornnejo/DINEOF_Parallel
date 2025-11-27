#!/bin/bash
# Script to run the parallelized DINEOF

# Set number of threads
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export LD_LIBRARY_PATH=/home/sebastian.cornejo/miniconda3/envs/dineof_env/lib:$LD_LIBRARY_PATH

# Run DINEOF
cd DINEOF_Parallel
./dineof dineof.init > dineof_parallel.log 2>&1 &
echo "DINEOF Parallel started with PID $!"
