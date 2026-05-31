#include "spmm_opt.h"

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
    d_light_rows = nullptr;
    d_heavy_rows = nullptr;

    if (feat_in == 32) {
        std::vector<int> h_ptr(num_v + 1);
        checkCudaErrors(cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> light_rows, heavy_rows;
        light_rows.reserve(num_v);
        heavy_rows.reserve(num_v);
        for (int i = 0; i < num_v; ++i) {
            int deg = h_ptr[i + 1] - h_ptr[i];
            if (deg > 256) heavy_rows.push_back(i);
            else light_rows.push_back(i);
        }

        num_light_rows = (int)light_rows.size();
        num_heavy_rows = (int)heavy_rows.size();
        if (num_heavy_rows < 128) {
            num_light_rows = num_v;
            num_heavy_rows = 0;
        }
        if (num_heavy_rows > 0) {
            checkCudaErrors(cudaMalloc2((void **)&d_light_rows, num_light_rows * sizeof(int)));
            checkCudaErrors(cudaMalloc2((void **)&d_heavy_rows, num_heavy_rows * sizeof(int)));
            checkCudaErrors(cudaMemcpy(d_light_rows, light_rows.data(), num_light_rows * sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_heavy_rows, heavy_rows.data(), num_heavy_rows * sizeof(int), cudaMemcpyHostToDevice));
        }
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in == 32) {
        if (num_heavy_rows > 0) {
            spmm_kernel32_rows<<<(num_light_rows * 32 + block.x - 1) / block.x, block>>>(d_ptr, d_idx, d_val, vin, vout, d_light_rows, num_light_rows);
            spmm_kernel32_heavy<<<num_heavy_rows, 512>>>(d_ptr, d_idx, d_val, vin, vout, d_heavy_rows);
        } else {
            spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else {
        spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }
}
