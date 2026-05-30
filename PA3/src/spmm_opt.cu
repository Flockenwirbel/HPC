#include "spmm_opt.h"

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
    // TODO: your code
    int BLOCK_SIZE = 256;
    block.x = BLOCK_SIZE;
    grid.x = (num_v * 32 + block.x - 1) / block.x;

}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}