source /home/spack/spack/share/spack/setup-env.sh
spack load ucx
export I_MPI_PMI_LIBRARY=/usr/lib/x86_64-linux-gnu/libpmi2.so

srun --mpi=pmi2 -n 8 ./main $1
