#include "spmm_opt.h"

#include <algorithm>
#include <vector>

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < INFEATURE; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        vout[tid * INFEATURE + j] = result;
    }
}

__global__ void spmm_kernel32_rows(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lid = threadIdx.x & 31;
    if (wid >= num_rows) return;

    int row = rows[wid];
    int begin = ptr[row], end = ptr[row + 1];
    float acc = 0.0f;
    for (int i = begin; i < end; ++i) {
        acc += val[i] * vin[idx[i] * 32 + lid];
    }
    vout[row * 32 + lid] = acc;
}

__global__ void spmm_kernel32_heavy(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows) {
    int row = rows[blockIdx.x];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int begin = ptr[row], end = ptr[row + 1];

    float acc = 0.0f;
    for (int i = begin + wid; i < end; i += 16) {
        acc += val[i] * vin[idx[i] * 32 + lid];
    }

    __shared__ float part[16][32];
    part[wid][lid] = acc;
    __syncthreads();

    if (wid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) sum += part[i][lid];
        vout[row * 32 + lid] = sum;
    }
}

__global__ void spmm_kernel256_rows(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lid = threadIdx.x & 31;
    if (wid >= num_rows) return;

    int row = rows[wid];
    int begin = ptr[row], end = ptr[row + 1];
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    for (int i = begin; i < end; ++i) {
        int col = idx[i];
        float v = val[i];
        float *Brow = vin + col * 256;
        acc0 += v * Brow[lid];
        acc1 += v * Brow[lid + 32];
        acc2 += v * Brow[lid + 64];
        acc3 += v * Brow[lid + 96];
        acc4 += v * Brow[lid + 128];
        acc5 += v * Brow[lid + 160];
        acc6 += v * Brow[lid + 192];
        acc7 += v * Brow[lid + 224];
    }
    vout[row * 256 + lid] = acc0;
    vout[row * 256 + lid + 32] = acc1;
    vout[row * 256 + lid + 64] = acc2;
    vout[row * 256 + lid + 96] = acc3;
    vout[row * 256 + lid + 128] = acc4;
    vout[row * 256 + lid + 160] = acc5;
    vout[row * 256 + lid + 192] = acc6;
    vout[row * 256 + lid + 224] = acc7;
}

__global__ void spmm_kernel256_heavy(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows) {
    int row = rows[blockIdx.x];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int begin = ptr[row], end = ptr[row + 1];

    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    for (int i = begin + wid; i < end; i += 16) {
        int col = idx[i];
        float v = val[i];
        const float *Brow = vin + col * 256;
        acc0 += v * Brow[lid];
        acc1 += v * Brow[lid + 32];
        acc2 += v * Brow[lid + 64];
        acc3 += v * Brow[lid + 96];
        acc4 += v * Brow[lid + 128];
        acc5 += v * Brow[lid + 160];
        acc6 += v * Brow[lid + 192];
        acc7 += v * Brow[lid + 224];
    }

    __shared__ float part[16][256];
    part[wid][lid] = acc0;
    part[wid][lid + 32] = acc1;
    part[wid][lid + 64] = acc2;
    part[wid][lid + 96] = acc3;
    part[wid][lid + 128] = acc4;
    part[wid][lid + 160] = acc5;
    part[wid][lid + 192] = acc6;
    part[wid][lid + 224] = acc7;
    __syncthreads();

    if (wid == 0) {
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        float sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sum0 += part[i][lid];
            sum1 += part[i][lid + 32];
            sum2 += part[i][lid + 64];
            sum3 += part[i][lid + 96];
            sum4 += part[i][lid + 128];
            sum5 += part[i][lid + 160];
            sum6 += part[i][lid + 192];
            sum7 += part[i][lid + 224];
        }
        vout[row * 256 + lid] = sum0;
        vout[row * 256 + lid + 32] = sum1;
        vout[row * 256 + lid + 64] = sum2;
        vout[row * 256 + lid + 96] = sum3;
        vout[row * 256 + lid + 128] = sum4;
        vout[row * 256 + lid + 160] = sum5;
        vout[row * 256 + lid + 192] = sum6;
        vout[row * 256 + lid + 224] = sum7;
    }
}

__global__ void gather_first_idx_kernel(int *ptr, int *idx, int *rows, int *first, int num_rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    int row = rows[tid];
    first[tid] = idx[ptr[row]];
}

__global__ void spmm_clear_rows32(float *vout, int *rows, int num_rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows * 32) return;
    int row = rows[tid >> 5];
    int lid = tid & 31;
    vout[row * 32 + lid] = 0.0f;
}

__global__ void spmm_clear_rows256(float *vout, int *rows, int num_rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows * 256) return;
    int row = rows[tid >> 8];
    int lid = tid & 255;
    vout[row * 256 + lid] = 0.0f;
}

__global__ void spmm_kernel256_hub(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_hubs) {
    int hub_idx = blockIdx.x >> 5;
    int sub = blockIdx.x & 31;
    if (hub_idx >= num_hubs) return;
    int row = rows[hub_idx];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int begin = ptr[row], end = ptr[row + 1];
    int deg = end - begin;
    int chunk = (deg + 31) >> 5;
    int sub_begin = begin + sub * chunk;
    int sub_end = (sub_begin + chunk < end) ? sub_begin + chunk : end;
    if (sub_begin >= end) return;

    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    for (int i = sub_begin + wid; i < sub_end; i += 16) {
        int col = idx[i];
        float v = val[i];
        const float *Brow = vin + col * 256;
        acc0 += v * Brow[lid];
        acc1 += v * Brow[lid + 32];
        acc2 += v * Brow[lid + 64];
        acc3 += v * Brow[lid + 96];
        acc4 += v * Brow[lid + 128];
        acc5 += v * Brow[lid + 160];
        acc6 += v * Brow[lid + 192];
        acc7 += v * Brow[lid + 224];
    }

    __shared__ float part[16][256];
    part[wid][lid] = acc0;
    part[wid][lid + 32] = acc1;
    part[wid][lid + 64] = acc2;
    part[wid][lid + 96] = acc3;
    part[wid][lid + 128] = acc4;
    part[wid][lid + 160] = acc5;
    part[wid][lid + 192] = acc6;
    part[wid][lid + 224] = acc7;
    __syncthreads();

    if (wid == 0) {
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        float sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sum0 += part[i][lid];
            sum1 += part[i][lid + 32];
            sum2 += part[i][lid + 64];
            sum3 += part[i][lid + 96];
            sum4 += part[i][lid + 128];
            sum5 += part[i][lid + 160];
            sum6 += part[i][lid + 192];
            sum7 += part[i][lid + 224];
        }
        int base = row * 256;
        atomicAdd(vout + base + lid, sum0);
        atomicAdd(vout + base + lid + 32, sum1);
        atomicAdd(vout + base + lid + 64, sum2);
        atomicAdd(vout + base + lid + 96, sum3);
        atomicAdd(vout + base + lid + 128, sum4);
        atomicAdd(vout + base + lid + 160, sum5);
        atomicAdd(vout + base + lid + 192, sum6);
        atomicAdd(vout + base + lid + 224, sum7);
    }
}

__global__ void spmm_kernel32_hub(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_hubs) {
    int hub_idx = blockIdx.x / 4;
    int sub = blockIdx.x % 4;
    if (hub_idx >= num_hubs) return;
    int row = rows[hub_idx];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int begin = ptr[row], end = ptr[row + 1];
    int deg = end - begin;
    int chunk = (deg + 3) / 4;
    int sub_begin = begin + sub * chunk;
    int sub_end = (sub_begin + chunk < end) ? sub_begin + chunk : end;
    if (sub_begin >= end) return;

    float acc = 0.0f;
    for (int i = sub_begin + wid; i < sub_end; i += 16) {
        acc += val[i] * vin[idx[i] * 32 + lid];
    }

    __shared__ float part[16][32];
    part[wid][lid] = acc;
    __syncthreads();

    if (wid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) sum += part[i][lid];
        atomicAdd(vout + row * 32 + lid, sum);
    }
}

__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lid = threadIdx.x & 31;
    if (wid >= num_v) return;

    int begin = ptr[wid], end = ptr[wid + 1];

    if (INFEATURE == 32) {
        float acc = 0.0f;
        for (int i = begin; i < end; ++i) {
            acc += val[i] * vin[idx[i] * 32 + lid];
        }
        vout[wid * 32 + lid] = acc;
    } else {
        float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
        for (int i = begin; i < end; ++i) {
            int col = idx[i];
            float v = val[i];
            float *Brow = vin + col * INFEATURE;
            acc0 += v * Brow[lid];
            acc1 += v * Brow[lid + 32];
            acc2 += v * Brow[lid + 64];
            acc3 += v * Brow[lid + 96];
            acc4 += v * Brow[lid + 128];
            acc5 += v * Brow[lid + 160];
            acc6 += v * Brow[lid + 192];
            acc7 += v * Brow[lid + 224];
        }
        vout[wid * INFEATURE + lid] = acc0;
        vout[wid * INFEATURE + lid + 32] = acc1;
        vout[wid * INFEATURE + lid + 64] = acc2;
        vout[wid * INFEATURE + lid + 96] = acc3;
        vout[wid * INFEATURE + lid + 128] = acc4;
        vout[wid * INFEATURE + lid + 160] = acc5;
        vout[wid * INFEATURE + lid + 192] = acc6;
        vout[wid * INFEATURE + lid + 224] = acc7;
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    int BLOCK_SIZE = 256;
    block.x = BLOCK_SIZE;
    grid.x = (num_v * 32 + block.x - 1) / block.x;

    num_light_rows = num_v;
    num_heavy_rows = 0;
    num_hub_rows = 0;
    d_light_rows = nullptr;
    d_heavy_rows = nullptr;
    d_hub_rows = nullptr;

    if (feat_in == 32 || feat_in == 256) {
        std::vector<int> h_ptr(num_v + 1);
        checkCudaErrors(cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> light_rows, heavy_rows, hub_rows;
        light_rows.reserve(num_v);
        heavy_rows.reserve(num_v);
        hub_rows.reserve(num_v);
        int heavy_threshold = (feat_in == 256) ? 16 : 256;
        int hub_threshold = 4096;
        for (int i = 0; i < num_v; ++i) {
            int deg = h_ptr[i + 1] - h_ptr[i];
            if (deg > hub_threshold) hub_rows.push_back(i);
            else if (deg > heavy_threshold) heavy_rows.push_back(i);
            else light_rows.push_back(i);
        }

        num_light_rows = (int)light_rows.size();
        num_heavy_rows = (int)heavy_rows.size();
        num_hub_rows = (int)hub_rows.size();
        if (num_heavy_rows + num_hub_rows < 128) {
            num_light_rows = num_v;
            num_heavy_rows = 0;
            num_hub_rows = 0;
        }
        if (num_heavy_rows + num_hub_rows > 0) {
            // Sort heavy rows by first column index for L2 locality
            auto sort_by_first_idx = [&](std::vector<int> &rows) {
                if (rows.empty()) return;
                int n = (int)rows.size();
                std::vector<std::pair<int,int>> order;
                order.reserve(n);
                std::vector<int> first(n);

                int *d_rows_tmp = nullptr;
                int *d_first = nullptr;
                checkCudaErrors(cudaMalloc((void **)&d_rows_tmp, n * sizeof(int)));
                checkCudaErrors(cudaMalloc((void **)&d_first, n * sizeof(int)));
                checkCudaErrors(cudaMemcpy(d_rows_tmp, rows.data(), n * sizeof(int), cudaMemcpyHostToDevice));
                gather_first_idx_kernel<<<(n + 255) / 256, 256>>>(d_ptr, d_idx, d_rows_tmp, d_first, n);
                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaMemcpy(first.data(), d_first, n * sizeof(int), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaFree(d_rows_tmp));
                checkCudaErrors(cudaFree(d_first));

                for (int i = 0; i < n; ++i) order.push_back({first[i], i});
                std::sort(order.begin(), order.end());
                std::vector<int> sorted;
                sorted.reserve(n);
                for (auto &p : order) sorted.push_back(rows[p.second]);
                rows.swap(sorted);
            };
            sort_by_first_idx(heavy_rows);
            sort_by_first_idx(hub_rows);

            if (num_light_rows > 0) {
                checkCudaErrors(cudaMalloc2((void **)&d_light_rows, num_light_rows * sizeof(int)));
                checkCudaErrors(cudaMemcpy(d_light_rows, light_rows.data(), num_light_rows * sizeof(int), cudaMemcpyHostToDevice));
            }
            if (num_heavy_rows > 0) {
                checkCudaErrors(cudaMalloc2((void **)&d_heavy_rows, num_heavy_rows * sizeof(int)));
                checkCudaErrors(cudaMemcpy(d_heavy_rows, heavy_rows.data(), num_heavy_rows * sizeof(int), cudaMemcpyHostToDevice));
            }
            if (num_hub_rows > 0) {
                checkCudaErrors(cudaMalloc2((void **)&d_hub_rows, num_hub_rows * sizeof(int)));
                checkCudaErrors(cudaMemcpy(d_hub_rows, hub_rows.data(), num_hub_rows * sizeof(int), cudaMemcpyHostToDevice));
            }
        }
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in == 32) {
        if (num_heavy_rows + num_hub_rows > 0) {
            if (num_light_rows > 0)
                spmm_kernel32_rows<<<(num_light_rows * 32 + block.x - 1) / block.x, block>>>(d_ptr, d_idx, d_val, vin, vout, d_light_rows, num_light_rows);
            if (num_heavy_rows > 0)
                spmm_kernel32_heavy<<<num_heavy_rows, 512>>>(d_ptr, d_idx, d_val, vin, vout, d_heavy_rows);
            if (num_hub_rows > 0) {
                spmm_clear_rows32<<<(num_hub_rows * 32 + block.x - 1) / block.x, block>>>(vout, d_hub_rows, num_hub_rows);
                spmm_kernel32_hub<<<num_hub_rows * 4, 512>>>(d_ptr, d_idx, d_val, vin, vout, d_hub_rows, num_hub_rows);
            }
        } else {
            spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else if (feat_in == 256) {
        if (num_heavy_rows + num_hub_rows > 0) {
            if (num_light_rows > 0)
                spmm_kernel256_rows<<<(num_light_rows * 32 + block.x - 1) / block.x, block>>>(d_ptr, d_idx, d_val, vin, vout, d_light_rows, num_light_rows);
            if (num_heavy_rows > 0)
                spmm_kernel256_heavy<<<num_heavy_rows, 512>>>(d_ptr, d_idx, d_val, vin, vout, d_heavy_rows);
            if (num_hub_rows > 0) {
                spmm_clear_rows256<<<(num_hub_rows * 256 + block.x - 1) / block.x, block>>>(vout, d_hub_rows, num_hub_rows);
                spmm_kernel256_hub<<<num_hub_rows * 32, 512>>>(d_ptr, d_idx, d_val, vin, vout, d_hub_rows, num_hub_rows);
            }
        } else {
            spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else {
        spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }
}
