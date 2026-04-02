#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort() {
	if (out_of_range) return;
	/** Your code ... */
	// you can use variables in class Worker: n, nprocs, rank, block_len, data

	// local sort
	float* local_recv_buf = new float[block_len];
	float* local_merged = new float[block_len * 2];
	std::sort(data, data + block_len);
	for (int phase = 0; phase < nprocs; ++phase) {
		int partner = (rank % 2 == phase % 2) ? rank + 1 : rank - 1; // determine partner for this phase
		if (partner >= 0 && partner < nprocs) {
            MPI_Sendrecv(data, block_len, MPI_FLOAT, partner, 0,
                         local_recv_buf, block_len, MPI_FLOAT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::merge(data, data + block_len, local_recv_buf, local_recv_buf + block_len, local_merged);

            if (rank < partner) std::copy(local_merged, local_merged + block_len, data);
            else std::copy(local_merged + block_len, local_merged + 2 * block_len, data);
        }
	}

	delete[] local_recv_buf;
	delete[] local_merged;
}
