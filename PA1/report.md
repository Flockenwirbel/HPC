# PA 1 Report

刘家豪 2024010779

## `sort()` 函数说明

源代码如下.

```cpp
void Worker::sort() {
	if (out_of_range) return; // if this process has no data, skip sorting

	// Calculate block size and active processes
	size_t std_block = ceiling(n, (size_t)nprocs);
	int activeprocs = (int)ceiling(n, std_block);

	// If only one process is active, just sort locally and return.
	if (activeprocs == 1) {
		radix_sort_float(data, block_len);
		return;
	}

	// Helper lambda to get the partner process for a given phase.
	auto get_partner = [this, activeprocs](int phase) {
		int p = (this->rank % 2 == phase % 2) ? this->rank + 1 : this->rank - 1;
		if (p < 0 || p >= activeprocs) return -1;
		return p;
	};

	// Helper lambda to get the actual block length for a given process rank.
	auto get_block_len = [&](int r) -> int {
		if (r < 0 || r >= activeprocs) return 0;
		size_t off = std_block * r;
		if (off >= n) return 0;
		return (int)std::min(std_block, n - off);
	};

	// local sort
	radix_sort_float(data, block_len);

	// Buffers for communication and merging
	float* swap_buf  = new float[block_len];
	float* recv_buf  = new float[std_block];
	float* cur  = data;
	float* aux  = swap_buf;

	MPI_Request reqs[2];

	for (int phase = 0; phase < activeprocs; ++phase) { // total activeprocs phases
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

	if (cur != data) // final copy back if the last merge result is in aux
		std::copy(cur, cur + block_len, data);

	delete[] swap_buf;
	delete[] recv_buf;
}
```

其中实现基数排序的辅助函数定义如下:

```cpp
// engineered to sort in ascending order by interpreting the bit pattern of the float.
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
	if (n <= 64) { std::sort(arr, arr + n); return; } // For small arrays, std::sort is faster than radix sort.

	// Convert floats to sortable integers
	uint32_t* keys = new uint32_t[n];
	uint32_t* buf  = new uint32_t[n];

	// Transform the float array into a sortable integer array.
	for (size_t i = 0; i < n; i++)
		keys[i] = float_to_sortable(arr[i]);

	// Perform radix sort on the integer keys, processing 8 bits at a time.
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
```
## 性能优化方式与结果

- 使用 `MPI_Irecv` 在当前轮计算时提前挂起下一轮的接收请求, 以实现通信和计算的重叠.
- 使用了 `recv_buf[2]` 两个接收缓冲区, 配合 `buf_idx` 和 `next_buf` 交替使用.
- 进行归并之前, 先比较本地边界值与对方边界值, 以快速判断是否需要归并, 从而避免不必要的内存拷贝和比较.
- 针对块长度相等的情况, 归并时不需要检查越界, 可以使用更简洁的循环.
- 使用 `std::swap` 在归并后交换缓冲区指针, 避免不必要的内存拷贝.
- 通过 `active_procs` 逻辑计算出真正持有数据的进程数, 避免了空进程参与通信和计算.
- 由于给定数据有界, 选择基数排序作为局部排序算法, 以获得更好的性能.

采取以上优化措施之后, 使用提供的 `100000000.dat` 进行测试, 性能对比如下.

| 进程数 | 优化前时间 (ms) | 优化后时间 (ms) | 优化后加速比 |
|--------|----------------|----------------|--------|
| $1\times 1$ | 12275.305 | 3526.510 | / |
| $1\times 2$ | 6758.324 | 2143.647 | 1.65x |
| $1\times 4$ | 3584.252 | 1338.911 | 2.63x |
| $1\times 8$ | 2020.664 | 913.647 | 3.86x |
| $1\times 16$ | 1274.377 | 732.856 | 4.81x |
| $2\times 16$ | 1080.625 | 553.517 | 6.37x |