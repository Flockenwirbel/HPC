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

	if (nprocs == 1) {
		std::sort(data, data + block_len);
		return;
	}

	auto get_partner = [this](int phase) {
        int p = (this->rank % 2 == phase % 2) ? this->rank + 1 : this->rank - 1;
        if (p < 0 || p >= (int)this->nprocs) return -1;
        return p;
    };

	// local sort
	std::sort(data, data + block_len);

	float* allocated_buf = new float[block_len]; 
	float* current_data = data;
	float* other_data = allocated_buf;

	// use 2 temporary recvbufs, alternatively for current merge and next recv
	float* recv_buf[2];
	recv_buf[0] = new float[block_len];
	recv_buf[1] = new float[block_len];
	
	MPI_Request send_req = MPI_REQUEST_NULL;
	MPI_Request recv_req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

	// pre-post recv request for the 0-th phase, to overlap communication with local sort
	int p_curr = get_partner(0);
	if (p_curr != -1) {
		MPI_Irecv(recv_buf[0], block_len, MPI_FLOAT, p_curr, 0, MPI_COMM_WORLD, &recv_req[0]);
	}

	for (int phase = 0; phase < nprocs; ++phase) {
		p_curr = get_partner(phase);
		int buf_idx = phase % 2;
		int next_buf = (phase + 1) % 2;
		int p_next = (phase + 1 < (int)nprocs) ? get_partner(phase + 1) : -1;

		// pre-post recv for next phase using next_buf as tag to avoid message mismatches
		if (p_next != -1) {
			MPI_Irecv(recv_buf[next_buf], block_len, MPI_FLOAT, p_next, next_buf, MPI_COMM_WORLD, &recv_req[next_buf]);
		}

		if (p_curr != -1) {
			// send using buf_idx as tag
			MPI_Isend(current_data, block_len, MPI_FLOAT, p_curr, buf_idx, MPI_COMM_WORLD, &send_req);
			MPI_Wait(&recv_req[buf_idx], MPI_STATUS_IGNORE);

			bool merged = false;

			if (rank < p_curr) { 
				// keep lower half
				if (current_data[block_len - 1] > recv_buf[buf_idx][0]) {
					int i = 0, j = 0, k = 0;
					while (k < (int)block_len) {
						if (current_data[i] <= recv_buf[buf_idx][j]) other_data[k++] = current_data[i++];
						else other_data[k++] = recv_buf[buf_idx][j++];
					}
					merged = true;
				}
			} else { 
				// keep upper half
				if (current_data[0] < recv_buf[buf_idx][block_len - 1]) {
					int i = (int)block_len - 1, j = (int)block_len - 1, k = (int)block_len - 1;
					while (k >= 0) {
						if (current_data[i] >= recv_buf[buf_idx][j]) other_data[k--] = current_data[i--];
						else other_data[k--] = recv_buf[buf_idx][j--];
					}
					merged = true;
				}
			}

			MPI_Wait(&send_req, MPI_STATUS_IGNORE);
			
			// only swap if merge actually occurred
			if (merged) std::swap(current_data, other_data);
		}
	}

	if (current_data != data) {
		std::copy(current_data, current_data + block_len, data);
	}

	delete[] allocated_buf;
	delete[] recv_buf[0];
	delete[] recv_buf[1];
}
