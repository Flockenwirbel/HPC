#include <stdint.h>
#include <math.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

constexpr int kOfficialHidx = 64;
constexpr int kOfficialDidx = 128;
constexpr int kOfficialPageSize = 64;
constexpr int kThreadsPerBlock = 256;
constexpr int kPage64ThreadsPerBlock = 512;
constexpr int kBf16PerUint4 = 8;
constexpr int kOfficialSharedVecCount = kOfficialHidx * kOfficialDidx / kBf16PerUint4;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__global__ void fill_neg_inf_kernel(
    __nv_bfloat16* __restrict__ scores,
    int64_t total
) {
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        scores[idx] = neg_inf;
    }
}

__global__ void dsa_indexer_scores_page_fallback_kernel(
    const __nv_bfloat16* __restrict__ q_idx,
    const __nv_bfloat16* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ w_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    __nv_bfloat16* __restrict__ scores,
    int64_t B,
    int64_t Hidx,
    int64_t Didx,
    int64_t MaxPages,
    int64_t PageSize
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t total_pages = B * MaxPages;
    const int64_t max_seq_len = MaxPages * PageSize;

    for (int64_t page_linear = blockIdx.x; page_linear < total_pages; page_linear += gridDim.x) {
        const int64_t b = page_linear / MaxPages;
        const int64_t logical_page = page_linear - b * MaxPages;
        const int64_t page_start = logical_page * PageSize;
        const int32_t visible = context_lens[b];

        if (page_start >= (int64_t)visible) {
            continue;
        }

        const int64_t remaining = (int64_t)visible - page_start;
        const int64_t valid_count = remaining < PageSize ? remaining : PageSize;
        const int32_t physical_page = block_table[b * MaxPages + logical_page];
        const int64_t q_batch_base = b * Hidx * Didx;
        const int64_t w_batch_base = b * Hidx;
        const int64_t page_k_base = (int64_t)physical_page * PageSize * Didx;

        for (int64_t token = warp; token < valid_count; token += warps_per_block) {
            const int64_t k_base = page_k_base + token * Didx;
            float acc = 0.0f;

            for (int64_t head = 0; head < Hidx; ++head) {
                const int64_t q_base = q_batch_base + head * Didx;
                float dot = 0.0f;
                for (int64_t d = lane; d < Didx; d += 32) {
                    dot = fmaf(__bfloat162float(q_idx[q_base + d]),
                               __bfloat162float(k_idx_cache[k_base + d]),
                               dot);
                }
                dot = warp_reduce_sum(dot);
                if (lane == 0 && dot > 0.0f) {
                    acc = fmaf(__bfloat162float(w_idx[w_batch_base + head]), dot, acc);
                }
            }

            if (lane == 0) {
                scores[b * max_seq_len + page_start + token] = __float2bfloat16(acc);
            }
        }
    }
}

__global__ void dsa_indexer_scores_page64_kernel(
    const __nv_bfloat16* __restrict__ q_idx,
    const __nv_bfloat16* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ w_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    __nv_bfloat16* __restrict__ scores,
    int64_t B,
    int64_t MaxPages
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int64_t total_pages = B * MaxPages;
    const int64_t max_seq_len = MaxPages * (int64_t)kOfficialPageSize;

    __shared__ uint4 q_shared_vec[kOfficialSharedVecCount];
    __shared__ uint4 k_shared_vec[kOfficialSharedVecCount];
    __shared__ float dot_shared[kOfficialHidx * kOfficialPageSize];
    __nv_bfloat16* q_shared = reinterpret_cast<__nv_bfloat16*>(q_shared_vec);
    __nv_bfloat16* k_shared = reinterpret_cast<__nv_bfloat16*>(k_shared_vec);

    for (int64_t page_linear = blockIdx.x; page_linear < total_pages; page_linear += gridDim.x) {
        const int64_t b = page_linear / MaxPages;
        const int64_t logical_page = page_linear - b * MaxPages;
        const int64_t page_start = logical_page * kOfficialPageSize;
        const int32_t visible = context_lens[b];

        const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);

        if (page_start >= (int64_t)visible) {
            if (tid < kOfficialPageSize) {
                scores[b * max_seq_len + page_start + tid] = neg_inf;
            }
        } else {
            const int64_t remaining = (int64_t)visible - page_start;
            const int valid_count = remaining < kOfficialPageSize ? (int)remaining : kOfficialPageSize;
            const int32_t physical_page = block_table[b * MaxPages + logical_page];

            const int64_t q_base = b * (int64_t)kOfficialHidx * kOfficialDidx;
            const uint4* q_global_vec = reinterpret_cast<const uint4*>(q_idx + q_base);
            for (int idx = tid; idx < kOfficialSharedVecCount; idx += blockDim.x) {
                q_shared_vec[idx] = q_global_vec[idx];
            }

            const int64_t w_base = b * (int64_t)kOfficialHidx;

            const int64_t k_base = (int64_t)physical_page * kOfficialPageSize * kOfficialDidx;
            const uint4* k_global_vec = reinterpret_cast<const uint4*>(k_idx_cache + k_base);
            for (int idx = tid; idx < kOfficialSharedVecCount; idx += blockDim.x) {
                k_shared_vec[idx] = k_global_vec[idx];
            }

            __syncthreads();

            if (warp < 16) {
                const int m_tile = warp >> 2;
                const int n_tile = warp & 3;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
                nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                #pragma unroll
                for (int k_tile = 0; k_tile < 8; ++k_tile) {
                    const __nv_bfloat16* a_ptr = q_shared + (m_tile * 16) * kOfficialDidx + k_tile * 16;
                    const __nv_bfloat16* b_ptr = k_shared + (n_tile * 16) * kOfficialDidx + k_tile * 16;
                    nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, kOfficialDidx);
                    nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, kOfficialDidx);
                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                float* c_ptr = dot_shared + (m_tile * 16) * kOfficialPageSize + n_tile * 16;
                nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, kOfficialPageSize, nvcuda::wmma::mem_row_major);
            }

            __syncthreads();

            if (tid < kOfficialPageSize) {
                if (tid < valid_count) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int head = 0; head < kOfficialHidx; ++head) {
                        const float dot = dot_shared[head * kOfficialPageSize + tid];
                        if (dot > 0.0f) {
                            acc = fmaf(__bfloat162float(w_idx[w_base + head]), dot, acc);
                        }
                    }
                    scores[b * max_seq_len + page_start + tid] = __float2bfloat16(acc);
                } else {
                    scores[b * max_seq_len + page_start + tid] = neg_inf;
                }
            }
        }

        __syncthreads();
    }
}

}  // namespace

extern "C" void run_kernel(
    const __nv_bfloat16* q_idx,
    const __nv_bfloat16* k_idx_cache,
    const __nv_bfloat16* w_idx,
    const int32_t* block_table,
    const int32_t* context_lens,
    __nv_bfloat16* scores,
    int64_t B,
    int64_t Hidx,
    int64_t Didx,
    int64_t MaxPages,
    int64_t PageSize
) {
    const int64_t max_seq_len = MaxPages * PageSize;
    const int64_t total_scores = B * max_seq_len;
    if (B <= 0 || Hidx <= 0 || Didx <= 0 || MaxPages <= 0 || PageSize <= 0 || total_scores <= 0) {
        return;
    }

    if (Hidx == kOfficialHidx && Didx == kOfficialDidx && PageSize == kOfficialPageSize) {
        int64_t page_blocks64 = B * MaxPages;
        if (page_blocks64 > 65535) {
            page_blocks64 = 65535;
        }
        dsa_indexer_scores_page64_kernel<<<(int)page_blocks64, kPage64ThreadsPerBlock>>>(
            q_idx,
            k_idx_cache,
            w_idx,
            block_table,
            context_lens,
            scores,
            B,
            MaxPages
        );
        return;
    }

    int64_t fill_blocks64 = (total_scores + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (fill_blocks64 > 65535) {
        fill_blocks64 = 65535;
    }
    fill_neg_inf_kernel<<<(int)fill_blocks64, kThreadsPerBlock>>>(scores, total_scores);

    int64_t page_blocks64 = B * MaxPages;
    if (page_blocks64 > 65535) {
        page_blocks64 = 65535;
    }
    dsa_indexer_scores_page_fallback_kernel<<<(int)page_blocks64, kThreadsPerBlock>>>(
        q_idx,
        k_idx_cache,
        w_idx,
        block_table,
        context_lens,
        scores,
        B,
        Hidx,
        Didx,
        MaxPages,
        PageSize
    );
}
