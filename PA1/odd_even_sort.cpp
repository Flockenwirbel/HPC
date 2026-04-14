#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "worker.h"

static inline uint32_t float_to_sortable(float f) {
	uint32_t u;
	std::memcpy(&u, &f, sizeof(u));
	uint32_t mask = (u >> 31) ? 0xFFFFFFFFu : 0x80000000u;
	return u ^ mask;
}

static inline float sortable_to_float(uint32_t u) {
	uint32_t mask = (u & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
	u ^= mask;
	float f;
	std::memcpy(&f, &u, sizeof(f));
	return f;
}

static void radix_sort_float(float* arr, size_t n) {
	if (n <= 64) { std::sort(arr, arr + n); return; }

	uint32_t* keys = new uint32_t[n];
	uint32_t* buf  = new uint32_t[n];

	for (size_t i = 0; i < n; i++)
		keys[i] = float_to_sortable(arr[i]);

	for (int shift = 0; shift < 32; shift += 8) {
		size_t cnt[256] = {};
		for (size_t i = 0; i < n; i++)
			cnt[(keys[i] >> shift) & 0xFF]++;

		size_t psum = 0;
		for (int b = 0; b < 256; b++) {
			size_t c = cnt[b];
			cnt[b] = psum;
			psum += c;
		}

		for (size_t i = 0; i < n; i++)
			buf[cnt[(keys[i] >> shift) & 0xFF]++] = keys[i];

		std::swap(keys, buf);
	}

	for (size_t i = 0; i < n; i++)
		arr[i] = sortable_to_float(keys[i]);

	delete[] keys;
	delete[] buf;
}

void Worker::sort() {
	if (out_of_range) return;

	size_t std_block = ceiling(n, (size_t)nprocs);
	int activeprocs = (int)ceiling(n, std_block);

	if (activeprocs == 1) {
		radix_sort_float(data, block_len);
		return;
	}

	auto get_partner = [this, activeprocs](int phase) {
		int p = (this->rank % 2 == phase % 2) ? this->rank + 1 : this->rank - 1;
		if (p < 0 || p >= activeprocs) return -1;
		return p;
	};

	auto get_block_len = [&](int r) -> int {
		if (r < 0 || r >= activeprocs) return 0;
		size_t off = std_block * r;
		if (off >= n) return 0;
		return (int)std::min(std_block, n - off);
	};

	// local sort
	radix_sort_float(data, block_len);

	float* swap_buf  = new float[block_len];
	float* recv_buf  = new float[std_block];
	float* cur  = data;
	float* aux  = swap_buf;

	MPI_Request reqs[2];

	for (int phase = 0; phase < activeprocs; ++phase) {
		int p = get_partner(phase);
		if (p == -1) continue;

		int pcl = get_block_len(p);
		int tag = phase & 1;

		MPI_Irecv(recv_buf, pcl, MPI_FLOAT, p, tag, MPI_COMM_WORLD, &reqs[0]);
		MPI_Isend(cur, (int)block_len, MPI_FLOAT, p, tag, MPI_COMM_WORLD, &reqs[1]);
		MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

		bool merged = false;

		if (rank < p) {
			// keep the lower block_len elements
			if (cur[block_len - 1] > recv_buf[0]) {
				int i = 0, j = 0, k = 0;
				while (k < (int)block_len) {
					if (j >= pcl || cur[i] <= recv_buf[j]) aux[k++] = cur[i++];
					else                                    aux[k++] = recv_buf[j++];
				}
				merged = true;
			}
		} else {
			// keep the upper block_len elements
			if (cur[0] < recv_buf[pcl - 1]) {
				int i = (int)block_len - 1, j = pcl - 1, k = (int)block_len - 1;
				while (k >= 0) {
					if      (i < 0)                        aux[k--] = recv_buf[j--];
					else if (j < 0 || cur[i] >= recv_buf[j]) aux[k--] = cur[i--];
					else                                    aux[k--] = recv_buf[j--];
				}
				merged = true;
			}
		}

		if (merged) std::swap(cur, aux);
	}

	if (cur != data)
		std::copy(cur, cur + block_len, data);

	delete[] swap_buf;
	delete[] recv_buf;
}
