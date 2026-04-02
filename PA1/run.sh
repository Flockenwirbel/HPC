#!/bin/bash

# 1 machine * 1 process
srun --nodes=1 -n 1 --cpu-bind=core ./odd_even_sort 100000000 /home/course/hpc/assignments/2026/data/public/PA1/100000000.dat

# 1 machine * 2 processes
srun --nodes=1 -n 2 --cpu-bind=core ./odd_even_sort 100000000 /home/course/hpc/assignments/2026/data/public/PA1/100000000.dat

# 1 machine * 4 processes
srun --nodes=1 -n 4 --cpu-bind=core ./odd_even_sort 100000000 /home/course/hpc/assignments/2026/data/public/PA1/100000000.dat

# 1 machine * 8 processes
srun --nodes=1 -n 8 --cpu-bind=core ./odd_even_sort 100000000 /home/course/hpc/assignments/2026/data/public/PA1/100000000.dat

# 1 machine * 16 processes
srun --nodes=1 -n 16 --cpu-bind=core ./odd_even_sort 100000000 /home/course/hpc/assignments/2026/data/public/PA1/100000000.dat

# 2 machines * 16 processes
srun --nodes=2 -n 32 --cpu-bind=core ./odd_even_sort 100000000 /home/course/hpc/assignments/2026/data/public/PA1/100000000.dat
