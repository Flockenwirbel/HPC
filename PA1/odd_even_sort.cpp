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

	// Compute the actual block_len for any rank (ceiling division, last rank may be shorter)
	size_t std_block = ceiling(n, (size_t)nprocs);
	auto get_block_len = [&](int r) -> int {
		if (r < 0 || r >= nprocs) return 0;
		size_t off = std_block * r;
		if (off >= n) return 0;
		return (int)std::min(std_block, n - off);
	};

	// local sort
	std::sort(data, data + block_len);

	float* allocated_buf = new float[block_len];
	float* current_data = data;
	float* other_data = allocated_buf;

	// recv bufs sized to max possible partner block (std_block)
	float* recv_buf[2];
	recv_buf[0] = new float[std_block];
	recv_buf[1] = new float[std_block];

	MPI_Request send_req = MPI_REQUEST_NULL;
	MPI_Request recv_req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

	// pre-post recv for phase 0 to overlap with local sort
	{
		int p0 = get_partner(0);
		if (p0 != -1)
			MPI_Irecv(recv_buf[0], get_block_len(p0), MPI_FLOAT, p0, 0, MPI_COMM_WORLD, &recv_req[0]);
	}

	for (int phase = 0; phase < nprocs; ++phase) {
		int p_curr = get_partner(phase);
		int buf_idx = phase % 2;
		int next_buf = (phase + 1) % 2;
		int p_next = (phase + 1 < nprocs) ? get_partner(phase + 1) : -1;

		// pre-post recv for next phase
		if (p_next != -1)
			MPI_Irecv(recv_buf[next_buf], get_block_len(p_next), MPI_FLOAT, p_next, next_buf, MPI_COMM_WORLD, &recv_req[next_buf]);

		if (p_curr != -1) {
			int pcl = get_block_len(p_curr);
			MPI_Isend(current_data, (int)block_len, MPI_FLOAT, p_curr, buf_idx, MPI_COMM_WORLD, &send_req);
			MPI_Wait(&recv_req[buf_idx], MPI_STATUS_IGNORE);

			bool merged = false;

			if (rank < p_curr) {
				// keep lower block_len elements
				if (current_data[block_len - 1] > recv_buf[buf_idx][0]) {
					int i = 0, j = 0, k = 0;
					if (pcl == (int)block_len) {
						// fast path: equal sizes, no bounds guard needed
						while (k < (int)block_len) {
							if (current_data[i] <= recv_buf[buf_idx][j]) other_data[k++] = current_data[i++];
							else other_data[k++] = recv_buf[buf_idx][j++];
						}
					} else {
						// slow path: partner (higher rank) has fewer elements
						while (k < (int)block_len) {
							if (j >= pcl || current_data[i] <= recv_buf[buf_idx][j]) other_data[k++] = current_data[i++];
							else other_data[k++] = recv_buf[buf_idx][j++];
						}
					}
					merged = true;
				}
			} else {
				// keep upper block_len elements
				if (current_data[0] < recv_buf[buf_idx][pcl - 1]) {
					int i = (int)block_len - 1, j = pcl - 1, k = (int)block_len - 1;
					if (pcl == (int)block_len) {
						// fast path: equal sizes
						while (k >= 0) {
							if (current_data[i] >= recv_buf[buf_idx][j]) other_data[k--] = current_data[i--];
							else other_data[k--] = recv_buf[buf_idx][j--];
						}
					} else {
						// slow path: we (higher rank) have more elements than partner
						while (k >= 0) {
							if (j < 0 || current_data[i] >= recv_buf[buf_idx][j]) other_data[k--] = current_data[i--];
							else other_data[k--] = recv_buf[buf_idx][j--];
						}
					}
					merged = true;
				}
			}

			MPI_Wait(&send_req, MPI_STATUS_IGNORE);

			if (merged) std::swap(current_data, other_data);
		}

		// Early termination: if no process merged this phase, the array is globally sorted.
		// Cancel the pre-posted recv for the next phase before breaking.
		int local_merged = (p_curr != -1 && /* merged flag captured below */ false) ? 1 : 0;
		// Re-check: use a separate flag outside the if block
		// (already have 'merged' in scope only inside; restructure below via global_merged)
		int global_merged;
		// 'merged' is declared inside the if block; hoist it via a phase-level variable
		// See restructured loop below — this placeholder is replaced in the next edit.
		(void)global_merged;
	}

	if (current_data != data) {
		std::copy(current_data, current_data + block_len, data);
	}

	delete[] allocated_buf;
	delete[] recv_buf[0];
	delete[] recv_buf[1];
}
