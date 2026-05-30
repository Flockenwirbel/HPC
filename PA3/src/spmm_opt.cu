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
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // warp id
    int lid = threadIdx.x & 31; // lane id
    if (wid >= num_v) return;

    int begin = ptr[wid], end = ptr[wid + 1];
    float result[8] = { 0.0f };
    for (int i = begin; i < end; ++i) {
        int idx_i = idx[i];
        float val_i = val[i];
        for (int j = lid, s = 0; j < INFEATURE; j += 32, ++s) {
            result[s] += vin[idx_i * INFEATURE + j] * val_i;
        }
    }
    for (int j = lid, s = 0; j < INFEATURE; j += 32, ++s) {
        vout[wid * INFEATURE + j] = result[s];
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    int BLOCK_SIZE = 128;
    block.x = BLOCK_SIZE;
    grid.x = (num_v * 32 + block.x - 1) / block.x;

}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}