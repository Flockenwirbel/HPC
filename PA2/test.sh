#!/bin/bash

# Load CUDA & GCC env
spack load cuda && spack load gcc@10.2.0

# Make
make clean && make

# Submit Job
for i in {1000,2500,5000,7500,10000}; do
    echo "Running benchmark with n = $i"
    srun -N 1 --gres=gpu:1 ./benchmark $i
done
