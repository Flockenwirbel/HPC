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

	// Handle special case: only 1 process, just sort locally
	if (nprocs == 1) {
		std::sort(data, data + block_len);
		return;
	}

	auto get_partner = [this](int phase) {
        int p = (this->rank % 2 == phase % 2) ? this->rank + 1 : this->rank - 1;
        return (p + (int)this->nprocs) % (int)this->nprocs;
    };

	// local sort
	std::sort(data, data + block_len);

	// use 2 pointers to avoid std::copy
	float* local_data = data;
	float* local_data_buf = new float[block_len];

	// use 2 temporary recvbufs, alternatively for current merge and next recv
	float* recv_buf[2];
	recv_buf[0] = new float[block_len];
	recv_buf[1] = new float[block_len];
	
	MPI_Request send_req = MPI_REQUEST_NULL;
	MPI_Request recv_req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

	// pre-post recv request for the 0-th phase, to overlap communication with local sort
	int p_curr = get_partner(0);
	MPI_Irecv(recv_buf[0], block_len, MPI_FLOAT, p_curr, 0, MPI_COMM_WORLD, &recv_req[0]);

	for (int phase = 0; phase < nprocs; ++phase) {
		p_curr = get_partner(phase);
		MPI_Isend(local_data, block_len, MPI_FLOAT, p_curr, 0, MPI_COMM_WORLD, &send_req);

		if (phase + 1 < (int)nprocs) {
			int p_next = get_partner(phase + 1);
			MPI_Irecv(recv_buf[(phase + 1) % 2], block_len, MPI_FLOAT, p_next, 0, MPI_COMM_WORLD, &recv_req[(phase + 1) % 2]); // pre-post recv for next phase
		}

		MPI_Wait(&recv_req[phase % 2], MPI_STATUS_IGNORE);

		// only merge half of the data required
		if (rank < p_curr) { // ->
			int i  = 0, j = 0, k = 0; // i for local_data, j for recv_buf, k for local_data_buf
			while (k < (int)block_len) {
				if (i < (int)block_len && (j == (int)block_len || local_data[i] <= recv_buf[phase % 2][j])) local_data_buf[k++] = local_data[i++];
				else local_data_buf[k++] = recv_buf[phase % 2][j++];
			}
		} else { // <-
			int i  = (int)block_len - 1, j = (int)block_len - 1, k = (int)block_len - 1;
			while (k >= 0) {
				if (i >= 0 && (j < 0 || local_data[i] >= recv_buf[phase % 2][j])) local_data_buf[k--] = local_data[i--];
				else local_data_buf[k--] = recv_buf[phase % 2][j--];
			}
		}

		MPI_Wait(&send_req, MPI_STATUS_IGNORE);
		std::swap(local_data, local_data_buf);
	}

	if (local_data != data) std::copy(local_data, local_data + block_len, data);

	delete[] local_data_buf;

	delete[] recv_buf[0];
	delete[] recv_buf[1];
}
