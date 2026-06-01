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

__global__ void spmm_kernel32_low4(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lid = threadIdx.x & 31;
    int slot = lid >> 3;
    int q = lid & 7;
    int rid = wid * 4 + slot;
    if (rid >= num_rows) return;

    int row = rows[rid];
    int begin = ptr[row], end = ptr[row + 1];
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 *vin4 = reinterpret_cast<const float4 *>(vin);
    for (int i = begin; i < end; ++i) {
        float v = val[i];
        float4 x = vin4[idx[i] * 8 + q];
        acc.x += v * x.x;
        acc.y += v * x.y;
        acc.z += v * x.z;
        acc.w += v * x.w;
    }
    reinterpret_cast<float4 *>(vout)[row * 8 + q] = acc;
}

__global__ void spmm_kernel32_rows(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lid = threadIdx.x & 31;
    if (wid >= num_rows) return;

    int row = rows[wid];
    int part = lid >> 3;
    int q = lid & 7;
    int begin = ptr[row], end = ptr[row + 1];
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 *vin4 = reinterpret_cast<const float4 *>(vin);
    for (int i = begin + part; i < end; i += 4) {
        float v = val[i];
        float4 x = vin4[idx[i] * 8 + q];
        acc.x += v * x.x;
        acc.y += v * x.y;
        acc.z += v * x.z;
        acc.w += v * x.w;
    }

    unsigned mask = 0xffffffffu;
    float x = acc.x, y = acc.y, z = acc.z, w = acc.w;
    x += __shfl_down_sync(mask, x, 16);
    y += __shfl_down_sync(mask, y, 16);
    z += __shfl_down_sync(mask, z, 16);
    w += __shfl_down_sync(mask, w, 16);
    x += __shfl_down_sync(mask, x, 8);
    y += __shfl_down_sync(mask, y, 8);
    z += __shfl_down_sync(mask, z, 8);
    w += __shfl_down_sync(mask, w, 8);
    if (lid < 8) {
        reinterpret_cast<float4 *>(vout)[row * 8 + q] = make_float4(x, y, z, w);
    }
}


__global__ void spmm_kernel32_rows8(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lid = threadIdx.x & 31;
    if (wid >= num_rows) return;

    int row = rows[wid];
    int part = lid >> 2;
    int q = lid & 3;
    int begin = ptr[row], end = ptr[row + 1];
    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 *vin4 = reinterpret_cast<const float4 *>(vin);
    for (int i = begin + part; i < end; i += 8) {
        float v = val[i];
        const float4 *Brow = vin4 + idx[i] * 8;
        float4 x0 = Brow[q];
        float4 x1 = Brow[q + 4];
        acc0.x += v * x0.x;
        acc0.y += v * x0.y;
        acc0.z += v * x0.z;
        acc0.w += v * x0.w;
        acc1.x += v * x1.x;
        acc1.y += v * x1.y;
        acc1.z += v * x1.z;
        acc1.w += v * x1.w;
    }

    unsigned mask = 0xffffffffu;
    float x0 = acc0.x, y0 = acc0.y, z0 = acc0.z, w0 = acc0.w;
    float x1 = acc1.x, y1 = acc1.y, z1 = acc1.z, w1 = acc1.w;
    x0 += __shfl_down_sync(mask, x0, 16);
    y0 += __shfl_down_sync(mask, y0, 16);
    z0 += __shfl_down_sync(mask, z0, 16);
    w0 += __shfl_down_sync(mask, w0, 16);
    x1 += __shfl_down_sync(mask, x1, 16);
    y1 += __shfl_down_sync(mask, y1, 16);
    z1 += __shfl_down_sync(mask, z1, 16);
    w1 += __shfl_down_sync(mask, w1, 16);
    x0 += __shfl_down_sync(mask, x0, 8);
    y0 += __shfl_down_sync(mask, y0, 8);
    z0 += __shfl_down_sync(mask, z0, 8);
    w0 += __shfl_down_sync(mask, w0, 8);
    x1 += __shfl_down_sync(mask, x1, 8);
    y1 += __shfl_down_sync(mask, y1, 8);
    z1 += __shfl_down_sync(mask, z1, 8);
    w1 += __shfl_down_sync(mask, w1, 8);
    x0 += __shfl_down_sync(mask, x0, 4);
    y0 += __shfl_down_sync(mask, y0, 4);
    z0 += __shfl_down_sync(mask, z0, 4);
    w0 += __shfl_down_sync(mask, w0, 4);
    x1 += __shfl_down_sync(mask, x1, 4);
    y1 += __shfl_down_sync(mask, y1, 4);
    z1 += __shfl_down_sync(mask, z1, 4);
    w1 += __shfl_down_sync(mask, w1, 4);
    if (lid < 4) {
        float4 *out4 = reinterpret_cast<float4 *>(vout) + row * 8;
        out4[q] = make_float4(x0, y0, z0, w0);
        out4[q + 4] = make_float4(x1, y1, z1, w1);
    }
}

__global__ void spmm_kernel32_rows16(int *ptr, int *idx, float *val, float *vin, float *vout, int *rows, int num_rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lid = threadIdx.x & 31;
    if (wid >= num_rows) return;

    int row = rows[wid];
    int part = lid >> 1;
    int q = lid & 1;
    int begin = ptr[row], end = ptr[row + 1];
    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 *vin4 = reinterpret_cast<const float4 *>(vin);
    for (int i = begin + part; i < end; i += 16) {
        float v = val[i];
        const float4 *Brow = vin4 + idx[i] * 8;
        float4 x0 = Brow[q];
        float4 x1 = Brow[q + 2];
        float4 x2 = Brow[q + 4];
        float4 x3 = Brow[q + 6];
        acc0.x += v * x0.x; acc0.y += v * x0.y; acc0.z += v * x0.z; acc0.w += v * x0.w;
        acc1.x += v * x1.x; acc1.y += v * x1.y; acc1.z += v * x1.z; acc1.w += v * x1.w;
        acc2.x += v * x2.x; acc2.y += v * x2.y; acc2.z += v * x2.z; acc2.w += v * x2.w;
        acc3.x += v * x3.x; acc3.y += v * x3.y; acc3.z += v * x3.z; acc3.w += v * x3.w;
    }

    unsigned mask = 0xffffffffu;
    float x0 = acc0.x, y0 = acc0.y, z0 = acc0.z, w0 = acc0.w;
    float x1 = acc1.x, y1 = acc1.y, z1 = acc1.z, w1 = acc1.w;
    float x2 = acc2.x, y2 = acc2.y, z2 = acc2.z, w2 = acc2.w;
    float x3 = acc3.x, y3 = acc3.y, z3 = acc3.z, w3 = acc3.w;
#define REDUCE_K32_SPLIT16(off) \
    x0 += __shfl_down_sync(mask, x0, off); y0 += __shfl_down_sync(mask, y0, off); z0 += __shfl_down_sync(mask, z0, off); w0 += __shfl_down_sync(mask, w0, off); \
    x1 += __shfl_down_sync(mask, x1, off); y1 += __shfl_down_sync(mask, y1, off); z1 += __shfl_down_sync(mask, z1, off); w1 += __shfl_down_sync(mask, w1, off); \
    x2 += __shfl_down_sync(mask, x2, off); y2 += __shfl_down_sync(mask, y2, off); z2 += __shfl_down_sync(mask, z2, off); w2 += __shfl_down_sync(mask, w2, off); \
    x3 += __shfl_down_sync(mask, x3, off); y3 += __shfl_down_sync(mask, y3, off); z3 += __shfl_down_sync(mask, z3, off); w3 += __shfl_down_sync(mask, w3, off)
    REDUCE_K32_SPLIT16(16);
    REDUCE_K32_SPLIT16(8);
    REDUCE_K32_SPLIT16(4);
    REDUCE_K32_SPLIT16(2);
#undef REDUCE_K32_SPLIT16

    if (lid < 2) {
        float4 *out4 = reinterpret_cast<float4 *>(vout) + row * 8;
        out4[q] = make_float4(x0, y0, z0, w0);
        out4[q + 2] = make_float4(x1, y1, z1, w1);
        out4[q + 4] = make_float4(x2, y2, z2, w2);
        out4[q + 6] = make_float4(x3, y3, z3, w3);
    }
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
    int hub_idx = blockIdx.x >> 4;
    int sub = blockIdx.x & 15;
    if (hub_idx >= num_hubs) return;
    int row = rows[hub_idx];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int begin = ptr[row], end = ptr[row + 1];
    int deg = end - begin;
    int chunk = (deg + 15) >> 4;
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
    num_low_rows = 0;
    num_heavy_rows = 0;
    num_hub_rows = 0;
    d_light_rows = nullptr;
    d_low_rows = nullptr;
    d_heavy_rows = nullptr;
    d_hub_rows = nullptr;
    use_split8_rows = false;
    use_split16_rows = false;

    if (feat_in == 32 || feat_in == 256) {
        std::vector<int> h_ptr(num_v + 1);
        checkCudaErrors(cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        int max_deg = 0;
        for (int i = 0; i < num_v; ++i) {
            max_deg = std::max(max_deg, h_ptr[i + 1] - h_ptr[i]);
        }
        float avg_deg = (num_v > 0) ? (float)num_e / (float)num_v : 0.0f;

        int heavy_threshold = 16;
        int hub_threshold = 4096;
        int low_threshold = -1;
        bool sort_light_rows = false;
        if (feat_in == 32) {
            low_threshold = 8;
            heavy_threshold = max_deg;  // K=32 uses split4 light rows plus hub rows; no separate heavy rows.
            if (avg_deg < 10.0f && max_deg > 8192) hub_threshold = 1024;
            else if (avg_deg > 100.0f && max_deg > 16000) hub_threshold = 32768;
            else hub_threshold = 8192;
            if (avg_deg > 8.0f && avg_deg < 20.0f && max_deg < 4096) low_threshold = 16;
            sort_light_rows = !(avg_deg > 400.0f && max_deg < 8192);
            use_split8_rows = (avg_deg < 8.0f) || (max_deg < 1000) || (avg_deg > 400.0f && max_deg < 8192);
            use_split16_rows = (avg_deg < 8.0f && max_deg > 10000 && max_deg < 20000);
        }

        std::vector<int> low_rows, light_rows, heavy_rows, hub_rows;
        low_rows.reserve(num_v);
        light_rows.reserve(num_v);
        heavy_rows.reserve(num_v);
        hub_rows.reserve(num_v);
        for (int i = 0; i < num_v; ++i) {
            int deg = h_ptr[i + 1] - h_ptr[i];
            if (deg > hub_threshold) hub_rows.push_back(i);
            else if (deg > heavy_threshold) heavy_rows.push_back(i);
            else if (feat_in == 32 && deg <= low_threshold) low_rows.push_back(i);
            else light_rows.push_back(i);
        }

        num_low_rows = (int)low_rows.size();
        num_light_rows = (int)light_rows.size();
        num_heavy_rows = (int)heavy_rows.size();
        num_hub_rows = (int)hub_rows.size();
        if (num_low_rows + num_heavy_rows + num_hub_rows < 128) {
            num_low_rows = 0;
            num_light_rows = num_v;
            num_heavy_rows = 0;
            num_hub_rows = 0;
        }
        if (num_low_rows + num_heavy_rows + num_hub_rows > 0) {
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
            if (sort_light_rows) sort_by_first_idx(light_rows);
            sort_by_first_idx(heavy_rows);
            sort_by_first_idx(hub_rows);

            if (num_low_rows > 0) {
                checkCudaErrors(cudaMalloc2((void **)&d_low_rows, num_low_rows * sizeof(int)));
                checkCudaErrors(cudaMemcpy(d_low_rows, low_rows.data(), num_low_rows * sizeof(int), cudaMemcpyHostToDevice));
            }
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
        if (num_low_rows + num_heavy_rows + num_hub_rows > 0) {
            if (num_low_rows > 0)
                spmm_kernel32_low4<<<(((num_low_rows + 3) / 4) * 32 + block.x - 1) / block.x, block>>>(d_ptr, d_idx, d_val, vin, vout, d_low_rows, num_low_rows);
            if (num_light_rows > 0) {
                if (use_split16_rows)
                    spmm_kernel32_rows16<<<(num_light_rows * 32 + block.x - 1) / block.x, block>>>(d_ptr, d_idx, d_val, vin, vout, d_light_rows, num_light_rows);
                else if (use_split8_rows)
                    spmm_kernel32_rows8<<<(num_light_rows * 32 + block.x - 1) / block.x, block>>>(d_ptr, d_idx, d_val, vin, vout, d_light_rows, num_light_rows);
                else
                    spmm_kernel32_rows<<<(num_light_rows * 32 + block.x - 1) / block.x, block>>>(d_ptr, d_idx, d_val, vin, vout, d_light_rows, num_light_rows);
            }
            if (num_heavy_rows > 0)
                spmm_kernel32_heavy<<<num_heavy_rows, 512>>>(d_ptr, d_idx, d_val, vin, vout, d_heavy_rows);
            if (num_hub_rows > 0) {
                spmm_clear_rows32<<<(num_hub_rows * 32 + block.x - 1) / block.x, block>>>(vout, d_hub_rows, num_hub_rows);
                spmm_kernel32_hub<<<num_hub_rows * 16, 512>>>(d_ptr, d_idx, d_val, vin, vout, d_hub_rows, num_hub_rows);
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
