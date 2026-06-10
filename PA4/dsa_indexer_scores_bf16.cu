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
constexpr int kPage64PartialStride = kOfficialPageSize + 1;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__device__ __forceinline__ float quad_group_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    return value;
}

__device__ __forceinline__ void store_bf16_pair(__nv_bfloat16* ptr, float x, float y) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = __floats2bfloat162_rn(x, y);
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

        const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);
        if (page_start >= (int64_t)visible) {
            for (int64_t token = tid; token < PageSize; token += blockDim.x) {
                scores[b * max_seq_len + page_start + token] = neg_inf;
            }
            continue;
        }

        const int64_t remaining = (int64_t)visible - page_start;
        const int64_t valid_count = remaining < PageSize ? remaining : PageSize;
        for (int64_t token = valid_count + tid; token < PageSize; token += blockDim.x) {
            scores[b * max_seq_len + page_start + token] = neg_inf;
        }
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
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t total_pages = B * MaxPages;
    const int64_t max_seq_len = MaxPages * (int64_t)kOfficialPageSize;

    __shared__ uint4 k_shared_vec[kOfficialSharedVecCount];
    __shared__ float partial_shared[4 * kPage64PartialStride];
    __shared__ float w_shared[kOfficialHidx];
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
            const int64_t w_base = b * (int64_t)kOfficialHidx;
            if (tid < kOfficialHidx) {
                w_shared[tid] = __bfloat162float(w_idx[w_base + tid]);
            }

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
                    const __nv_bfloat16* a_ptr = q_idx + q_base + (m_tile * 16) * kOfficialDidx + k_tile * 16;
                    const __nv_bfloat16* b_ptr = k_shared + (n_tile * 16) * kOfficialDidx + k_tile * 16;
                    nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, kOfficialDidx);
                    nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, kOfficialDidx);
                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                const int group = lane >> 2;
                const int quad = lane & 3;
                const float w0 = w_shared[m_tile * 16 + group];
                const float w1 = w_shared[m_tile * 16 + group + 8];

                float p0 = 0.0f;
                float p1 = 0.0f;
                float p2 = 0.0f;
                float p3 = 0.0f;
                p0 = fmaxf(c_frag.x[0], 0.0f) * w0;
                p1 = fmaxf(c_frag.x[1], 0.0f) * w0;
                p0 = fmaf(fmaxf(c_frag.x[2], 0.0f), w1, p0);
                p1 = fmaf(fmaxf(c_frag.x[3], 0.0f), w1, p1);
                p2 = fmaxf(c_frag.x[4], 0.0f) * w0;
                p3 = fmaxf(c_frag.x[5], 0.0f) * w0;
                p2 = fmaf(fmaxf(c_frag.x[6], 0.0f), w1, p2);
                p3 = fmaf(fmaxf(c_frag.x[7], 0.0f), w1, p3);

                p0 = quad_group_reduce_sum(p0);
                p1 = quad_group_reduce_sum(p1);
                p2 = quad_group_reduce_sum(p2);
                p3 = quad_group_reduce_sum(p3);
                if (group == 0) {
                    partial_shared[m_tile * kPage64PartialStride + n_tile * 16 + quad * 2] = p0;
                    partial_shared[m_tile * kPage64PartialStride + n_tile * 16 + quad * 2 + 1] = p1;
                    partial_shared[m_tile * kPage64PartialStride + n_tile * 16 + quad * 2 + 8] = p2;
                    partial_shared[m_tile * kPage64PartialStride + n_tile * 16 + quad * 2 + 9] = p3;
                }
            }

            __syncthreads();

            if (tid < kOfficialPageSize) {
                if (tid < valid_count) {
                    float acc = partial_shared[tid];
                    acc += partial_shared[kPage64PartialStride + tid];
                    acc += partial_shared[2 * kPage64PartialStride + tid];
                    acc += partial_shared[3 * kPage64PartialStride + tid];
                    scores[b * max_seq_len + page_start + tid] = __float2bfloat16(acc);
                } else {
                    scores[b * max_seq_len + page_start + tid] = neg_inf;
                }
            }
        }

        __syncthreads();
    }
}

__global__ void dsa_indexer_scores_page64_group4_parallel_kernel(
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
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int page_slot = warp >> 2;
    const int n_tile = warp & 3;
    const int64_t max_seq_len = MaxPages * (int64_t)kOfficialPageSize;
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);

    __shared__ uint4 q_shared_vec[kOfficialSharedVecCount];
    __shared__ float w_shared[kOfficialHidx];
    __nv_bfloat16* q_shared = reinterpret_cast<__nv_bfloat16*>(q_shared_vec);

    if ((int64_t)blockIdx.y >= B) {
        return;
    }
    const int64_t b = blockIdx.y;
    const int64_t group_in_batch = blockIdx.x;
    const int64_t first_logical_page = group_in_batch << 2;
    const int32_t visible = __ldg(context_lens + b);
        const bool full_group =
            first_logical_page + 3 < MaxPages &&
            (first_logical_page + 4) * (int64_t)kOfficialPageSize <= (int64_t)visible;

    if (first_logical_page * (int64_t)kOfficialPageSize >= (int64_t)visible) {
        if (tid < 4 * kOfficialPageSize) {
            const int store_page_slot = tid >> 6;
            const int token = tid & 63;
            const int64_t logical_page = first_logical_page + store_page_slot;
            if (logical_page < MaxPages) {
                scores[b * max_seq_len + logical_page * kOfficialPageSize + token] = neg_inf;
            }
        }
        return;
    }

        const int64_t q_base = b * (int64_t)kOfficialHidx * kOfficialDidx;
        const uint4* q_global_vec = reinterpret_cast<const uint4*>(q_idx + q_base);
        for (int idx = tid; idx < kOfficialSharedVecCount; idx += blockDim.x) {
            q_shared_vec[idx] = q_global_vec[idx];
        }

        const int64_t w_base = b * (int64_t)kOfficialHidx;
        if (tid < kOfficialHidx) {
            w_shared[tid] = __bfloat162float(w_idx[w_base + tid]);
        }

        if (!full_group && tid < 4 * kOfficialPageSize) {
            const int store_page_slot = tid >> 6;
            const int token = tid & 63;
            const int64_t logical_page = first_logical_page + store_page_slot;
            if (logical_page < MaxPages) {
                const int64_t store_page_start = logical_page * kOfficialPageSize;
                if (store_page_start + token >= (int64_t)visible) {
                    scores[b * max_seq_len + store_page_start + token] = neg_inf;
                }
            }
        }

        __syncthreads();

        const int64_t logical_page = first_logical_page + page_slot;
        const int64_t page_start = logical_page * kOfficialPageSize;
        if (logical_page < MaxPages && page_start < (int64_t)visible) {
            const int valid_count = full_group
                ? kOfficialPageSize
                : (((int64_t)visible - page_start) < kOfficialPageSize
                    ? (int)((int64_t)visible - page_start)
                    : kOfficialPageSize);
            const int32_t physical_page = __ldg(block_table + b * MaxPages + logical_page);
            const int64_t k_base = (int64_t)physical_page * kOfficialPageSize * kOfficialDidx;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c0_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c1_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c2_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c3_frag;
            nvcuda::wmma::fill_fragment(c0_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c1_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c2_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c3_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < 8; ++k_tile) {
                const __nv_bfloat16* b_ptr = k_idx_cache + k_base + (n_tile * 16) * kOfficialDidx + k_tile * 16;
                nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, kOfficialDidx);

                const __nv_bfloat16* a0_ptr = q_shared + k_tile * 16;
                nvcuda::wmma::load_matrix_sync(a_frag, a0_ptr, kOfficialDidx);
                nvcuda::wmma::mma_sync(c0_frag, a_frag, b_frag, c0_frag);

                const __nv_bfloat16* a1_ptr = q_shared + 16 * kOfficialDidx + k_tile * 16;
                nvcuda::wmma::load_matrix_sync(a_frag, a1_ptr, kOfficialDidx);
                nvcuda::wmma::mma_sync(c1_frag, a_frag, b_frag, c1_frag);

                const __nv_bfloat16* a2_ptr = q_shared + 32 * kOfficialDidx + k_tile * 16;
                nvcuda::wmma::load_matrix_sync(a_frag, a2_ptr, kOfficialDidx);
                nvcuda::wmma::mma_sync(c2_frag, a_frag, b_frag, c2_frag);

                const __nv_bfloat16* a3_ptr = q_shared + 48 * kOfficialDidx + k_tile * 16;
                nvcuda::wmma::load_matrix_sync(a_frag, a3_ptr, kOfficialDidx);
                nvcuda::wmma::mma_sync(c3_frag, a_frag, b_frag, c3_frag);
            }

            const int group = lane >> 2;
            const int quad = lane & 3;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

#define ACCUM_FRAGMENT_LOCAL(FRAG, MTILE) \
            do { \
                const float w0 = w_shared[(MTILE) * 16 + group]; \
                const float w1 = w_shared[(MTILE) * 16 + group + 8]; \
                float p0 = fmaxf((FRAG).x[0], 0.0f) * w0; \
                float p1 = fmaxf((FRAG).x[1], 0.0f) * w0; \
                p0 = fmaf(fmaxf((FRAG).x[2], 0.0f), w1, p0); \
                p1 = fmaf(fmaxf((FRAG).x[3], 0.0f), w1, p1); \
                float p2 = fmaxf((FRAG).x[4], 0.0f) * w0; \
                float p3 = fmaxf((FRAG).x[5], 0.0f) * w0; \
                p2 = fmaf(fmaxf((FRAG).x[6], 0.0f), w1, p2); \
                p3 = fmaf(fmaxf((FRAG).x[7], 0.0f), w1, p3); \
                acc0 += p0; \
                acc1 += p1; \
                acc2 += p2; \
                acc3 += p3; \
            } while (0)
            ACCUM_FRAGMENT_LOCAL(c0_frag, 0);
            ACCUM_FRAGMENT_LOCAL(c1_frag, 1);
            ACCUM_FRAGMENT_LOCAL(c2_frag, 2);
            ACCUM_FRAGMENT_LOCAL(c3_frag, 3);
#undef ACCUM_FRAGMENT_LOCAL
            acc0 = quad_group_reduce_sum(acc0);
            acc1 = quad_group_reduce_sum(acc1);
            acc2 = quad_group_reduce_sum(acc2);
            acc3 = quad_group_reduce_sum(acc3);

            if (group == 0) {
                const int token0 = n_tile * 16 + quad * 2;
                const int token1 = token0 + 1;
                const int token2 = token0 + 8;
                const int token3 = token0 + 9;
                __nv_bfloat16* out_ptr = scores + b * max_seq_len + page_start;
                if (full_group) {
                    store_bf16_pair(out_ptr + token0, acc0, acc1);
                    store_bf16_pair(out_ptr + token2, acc2, acc3);
                } else {
                    if (token1 < valid_count) {
                        store_bf16_pair(out_ptr + token0, acc0, acc1);
                    } else if (token0 < valid_count) {
                        out_ptr[token0] = __float2bfloat16(acc0);
                    }
                    if (token3 < valid_count) {
                        store_bf16_pair(out_ptr + token2, acc2, acc3);
                    } else if (token2 < valid_count) {
                        out_ptr[token2] = __float2bfloat16(acc2);
                    }
                }
            }
        }
}

__global__ void dsa_indexer_scores_page64_group8_pair_kernel(
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
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int pair_slot = warp >> 2;
    const int n_tile = warp & 3;
    const int64_t max_seq_len = MaxPages * (int64_t)kOfficialPageSize;
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);

    __shared__ uint4 q_shared_vec[kOfficialSharedVecCount];
    __shared__ float w_shared[kOfficialHidx];
    __nv_bfloat16* q_shared = reinterpret_cast<__nv_bfloat16*>(q_shared_vec);

    if ((int64_t)blockIdx.y >= B) {
        return;
    }
    const int64_t b = blockIdx.y;
    const int64_t group_in_batch = blockIdx.x;
    const int64_t first_logical_page = group_in_batch << 3;
    const int32_t visible = __ldg(context_lens + b);
        const bool full_group =
            first_logical_page + 7 < MaxPages &&
            (first_logical_page + 8) * (int64_t)kOfficialPageSize <= (int64_t)visible;

    if (first_logical_page * (int64_t)kOfficialPageSize >= (int64_t)visible) {
        for (int idx = tid; idx < 8 * kOfficialPageSize; idx += blockDim.x) {
            const int64_t logical_page = first_logical_page + (idx >> 6);
            if (logical_page < MaxPages) {
                scores[b * max_seq_len + logical_page * kOfficialPageSize + (idx & 63)] = neg_inf;
            }
        }
        return;
    }

        const int64_t q_base = b * (int64_t)kOfficialHidx * kOfficialDidx;
        const uint4* q_global_vec = reinterpret_cast<const uint4*>(q_idx + q_base);
        for (int idx = tid; idx < kOfficialSharedVecCount; idx += blockDim.x) {
            q_shared_vec[idx] = q_global_vec[idx];
        }

        const int64_t w_base = b * (int64_t)kOfficialHidx;
        if (tid < kOfficialHidx) {
            w_shared[tid] = __bfloat162float(w_idx[w_base + tid]);
        }

        if (!full_group) {
            for (int idx = tid; idx < 8 * kOfficialPageSize; idx += blockDim.x) {
                const int64_t logical_page = first_logical_page + (idx >> 6);
                if (logical_page < MaxPages) {
                    const int64_t s = logical_page * (int64_t)kOfficialPageSize + (idx & 63);
                    if (s >= (int64_t)visible) {
                        scores[b * max_seq_len + s] = neg_inf;
                    }
                }
            }
        }

        __syncthreads();

        const int64_t logical_page0 = first_logical_page + pair_slot * 2;
        const int64_t logical_page1 = logical_page0 + 1;
        const int64_t page_start0 = logical_page0 * (int64_t)kOfficialPageSize;
        const int64_t page_start1 = page_start0 + kOfficialPageSize;
        const bool page0_valid = logical_page0 < MaxPages && page_start0 < (int64_t)visible;
        const bool page1_valid = logical_page1 < MaxPages && page_start1 < (int64_t)visible;

        if (page0_valid || page1_valid) {
            const int valid_count0 = full_group ? kOfficialPageSize :
                (((int64_t)visible - page_start0) < kOfficialPageSize ?
                    (int)((int64_t)visible - page_start0) : kOfficialPageSize);
            const int valid_count1 = full_group ? kOfficialPageSize :
                (((int64_t)visible - page_start1) < kOfficialPageSize ?
                    (int)((int64_t)visible - page_start1) : kOfficialPageSize);
            const int32_t physical_page0 = page0_valid ? __ldg(block_table + b * MaxPages + logical_page0) : 0;
            const int32_t physical_page1 = page1_valid ? __ldg(block_table + b * MaxPages + logical_page1) : 0;
            const int64_t k_base0 = (int64_t)physical_page0 * kOfficialPageSize * kOfficialDidx;
            const int64_t k_base1 = (int64_t)physical_page1 * kOfficialPageSize * kOfficialDidx;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b0_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b1_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c00_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c01_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c02_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c03_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c10_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c11_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c12_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c13_frag;
            nvcuda::wmma::fill_fragment(c00_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c01_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c02_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c03_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c10_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c11_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c12_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c13_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < 8; ++k_tile) {
                if (page0_valid) {
                    const __nv_bfloat16* b0_ptr = k_idx_cache + k_base0 + (n_tile * 16) * kOfficialDidx + k_tile * 16;
                    nvcuda::wmma::load_matrix_sync(b0_frag, b0_ptr, kOfficialDidx);
                }
                if (page1_valid) {
                    const __nv_bfloat16* b1_ptr = k_idx_cache + k_base1 + (n_tile * 16) * kOfficialDidx + k_tile * 16;
                    nvcuda::wmma::load_matrix_sync(b1_frag, b1_ptr, kOfficialDidx);
                }

#define DO_PAIR_MMA(MTILE, C0, C1) \
                do { \
                    const __nv_bfloat16* a_ptr = q_shared + (MTILE) * 16 * kOfficialDidx + k_tile * 16; \
                    nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, kOfficialDidx); \
                    if (page0_valid) nvcuda::wmma::mma_sync((C0), a_frag, b0_frag, (C0)); \
                    if (page1_valid) nvcuda::wmma::mma_sync((C1), a_frag, b1_frag, (C1)); \
                } while (0)
                DO_PAIR_MMA(0, c00_frag, c10_frag);
                DO_PAIR_MMA(1, c01_frag, c11_frag);
                DO_PAIR_MMA(2, c02_frag, c12_frag);
                DO_PAIR_MMA(3, c03_frag, c13_frag);
#undef DO_PAIR_MMA
            }

            const int group = lane >> 2;
            const int quad = lane & 3;

#define ACCUM_FRAGMENT_PAIR_LOCAL(FRAG, MTILE, A0, A1, A2, A3) \
            do { \
                const float w0 = w_shared[(MTILE) * 16 + group]; \
                const float w1 = w_shared[(MTILE) * 16 + group + 8]; \
                float p0 = fmaxf((FRAG).x[0], 0.0f) * w0; \
                float p1 = fmaxf((FRAG).x[1], 0.0f) * w0; \
                p0 = fmaf(fmaxf((FRAG).x[2], 0.0f), w1, p0); \
                p1 = fmaf(fmaxf((FRAG).x[3], 0.0f), w1, p1); \
                float p2 = fmaxf((FRAG).x[4], 0.0f) * w0; \
                float p3 = fmaxf((FRAG).x[5], 0.0f) * w0; \
                p2 = fmaf(fmaxf((FRAG).x[6], 0.0f), w1, p2); \
                p3 = fmaf(fmaxf((FRAG).x[7], 0.0f), w1, p3); \
                (A0) += p0; \
                (A1) += p1; \
                (A2) += p2; \
                (A3) += p3; \
            } while (0)

            if (page0_valid) {
                float acc00 = 0.0f;
                float acc01 = 0.0f;
                float acc02 = 0.0f;
                float acc03 = 0.0f;
                ACCUM_FRAGMENT_PAIR_LOCAL(c00_frag, 0, acc00, acc01, acc02, acc03);
                ACCUM_FRAGMENT_PAIR_LOCAL(c01_frag, 1, acc00, acc01, acc02, acc03);
                ACCUM_FRAGMENT_PAIR_LOCAL(c02_frag, 2, acc00, acc01, acc02, acc03);
                ACCUM_FRAGMENT_PAIR_LOCAL(c03_frag, 3, acc00, acc01, acc02, acc03);
                acc00 = quad_group_reduce_sum(acc00);
                acc01 = quad_group_reduce_sum(acc01);
                acc02 = quad_group_reduce_sum(acc02);
                acc03 = quad_group_reduce_sum(acc03);
                if (group == 0) {
                    const int token0 = n_tile * 16 + quad * 2;
                    const int token1 = token0 + 1;
                    const int token2 = token0 + 8;
                    const int token3 = token0 + 9;
                    __nv_bfloat16* out_ptr = scores + b * max_seq_len + page_start0;
                    if (full_group) {
                        store_bf16_pair(out_ptr + token0, acc00, acc01);
                        store_bf16_pair(out_ptr + token2, acc02, acc03);
                    } else {
                        if (token1 < valid_count0) {
                            store_bf16_pair(out_ptr + token0, acc00, acc01);
                        } else if (token0 < valid_count0) {
                            out_ptr[token0] = __float2bfloat16(acc00);
                        }
                        if (token3 < valid_count0) {
                            store_bf16_pair(out_ptr + token2, acc02, acc03);
                        } else if (token2 < valid_count0) {
                            out_ptr[token2] = __float2bfloat16(acc02);
                        }
                    }
                }
            }

            if (page1_valid) {
                float acc10 = 0.0f;
                float acc11 = 0.0f;
                float acc12 = 0.0f;
                float acc13 = 0.0f;
                ACCUM_FRAGMENT_PAIR_LOCAL(c10_frag, 0, acc10, acc11, acc12, acc13);
                ACCUM_FRAGMENT_PAIR_LOCAL(c11_frag, 1, acc10, acc11, acc12, acc13);
                ACCUM_FRAGMENT_PAIR_LOCAL(c12_frag, 2, acc10, acc11, acc12, acc13);
                ACCUM_FRAGMENT_PAIR_LOCAL(c13_frag, 3, acc10, acc11, acc12, acc13);
                acc10 = quad_group_reduce_sum(acc10);
                acc11 = quad_group_reduce_sum(acc11);
                acc12 = quad_group_reduce_sum(acc12);
                acc13 = quad_group_reduce_sum(acc13);
                if (group == 0) {
                    const int token0 = n_tile * 16 + quad * 2;
                    const int token1 = token0 + 1;
                    const int token2 = token0 + 8;
                    const int token3 = token0 + 9;
                    __nv_bfloat16* out_ptr = scores + b * max_seq_len + page_start1;
                    if (full_group) {
                        store_bf16_pair(out_ptr + token0, acc10, acc11);
                        store_bf16_pair(out_ptr + token2, acc12, acc13);
                    } else {
                        if (token1 < valid_count1) {
                            store_bf16_pair(out_ptr + token0, acc10, acc11);
                        } else if (token0 < valid_count1) {
                            out_ptr[token0] = __float2bfloat16(acc10);
                        }
                        if (token3 < valid_count1) {
                            store_bf16_pair(out_ptr + token2, acc12, acc13);
                        } else if (token2 < valid_count1) {
                            out_ptr[token2] = __float2bfloat16(acc12);
                        }
                    }
                }
            }
#undef ACCUM_FRAGMENT_PAIR_LOCAL
        }
}

__global__ void dsa_indexer_scores_page64_group32_pair_kernel(
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
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int pair_slot = warp >> 2;
    const int n_tile = warp & 3;
    const int64_t max_seq_len = MaxPages * (int64_t)kOfficialPageSize;
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);

    __shared__ uint4 q_shared_vec[kOfficialSharedVecCount];
    __shared__ float w_shared[kOfficialHidx];
    __nv_bfloat16* q_shared = reinterpret_cast<__nv_bfloat16*>(q_shared_vec);

    const int64_t b = blockIdx.y;
    const int64_t first_logical_page = (int64_t)blockIdx.x << 5;
    const int32_t visible = __ldg(context_lens + b);
    const bool full_group =
        first_logical_page + 31 < MaxPages &&
        (first_logical_page + 32) * (int64_t)kOfficialPageSize <= (int64_t)visible;

    if (first_logical_page * (int64_t)kOfficialPageSize >= (int64_t)visible) {
        for (int idx = tid; idx < 32 * kOfficialPageSize; idx += blockDim.x) {
            const int64_t logical_page = first_logical_page + (idx >> 6);
            if (logical_page < MaxPages) {
                scores[b * max_seq_len + logical_page * kOfficialPageSize + (idx & 63)] = neg_inf;
            }
        }
        return;
    }

    const int64_t q_base = b * (int64_t)kOfficialHidx * kOfficialDidx;
    const uint4* q_global_vec = reinterpret_cast<const uint4*>(q_idx + q_base);
    for (int idx = tid; idx < kOfficialSharedVecCount; idx += blockDim.x) {
        q_shared_vec[idx] = q_global_vec[idx];
    }

    const int64_t w_base = b * (int64_t)kOfficialHidx;
    if (tid < kOfficialHidx) {
        w_shared[tid] = __bfloat162float(w_idx[w_base + tid]);
    }

    if (!full_group) {
        for (int idx = tid; idx < 32 * kOfficialPageSize; idx += blockDim.x) {
            const int64_t logical_page = first_logical_page + (idx >> 6);
            if (logical_page < MaxPages) {
                const int64_t s = logical_page * (int64_t)kOfficialPageSize + (idx & 63);
                if (s >= (int64_t)visible) {
                    scores[b * max_seq_len + s] = neg_inf;
                }
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int wave = 0; wave < 4; ++wave) {
        const int64_t logical_page0 = first_logical_page + wave * 8 + pair_slot * 2;
        const int64_t logical_page1 = logical_page0 + 1;
        const int64_t page_start0 = logical_page0 * (int64_t)kOfficialPageSize;
        const int64_t page_start1 = page_start0 + kOfficialPageSize;
        const bool page0_valid = logical_page0 < MaxPages && page_start0 < (int64_t)visible;
        const bool page1_valid = logical_page1 < MaxPages && page_start1 < (int64_t)visible;

        if (page0_valid || page1_valid) {
            const int valid_count0 = full_group ? kOfficialPageSize :
                (((int64_t)visible - page_start0) < kOfficialPageSize ?
                    (int)((int64_t)visible - page_start0) : kOfficialPageSize);
            const int valid_count1 = full_group ? kOfficialPageSize :
                (((int64_t)visible - page_start1) < kOfficialPageSize ?
                    (int)((int64_t)visible - page_start1) : kOfficialPageSize);
            const int32_t physical_page0 = page0_valid ? __ldg(block_table + b * MaxPages + logical_page0) : 0;
            const int32_t physical_page1 = page1_valid ? __ldg(block_table + b * MaxPages + logical_page1) : 0;
            const int64_t k_base0 = (int64_t)physical_page0 * kOfficialPageSize * kOfficialDidx;
            const int64_t k_base1 = (int64_t)physical_page1 * kOfficialPageSize * kOfficialDidx;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b0_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b1_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c00_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c01_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c02_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c03_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c10_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c11_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c12_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c13_frag;
            nvcuda::wmma::fill_fragment(c00_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c01_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c02_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c03_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c10_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c11_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c12_frag, 0.0f);
            nvcuda::wmma::fill_fragment(c13_frag, 0.0f);

#pragma unroll
            for (int k_tile = 0; k_tile < 8; ++k_tile) {
                if (page0_valid) {
                    const __nv_bfloat16* b0_ptr = k_idx_cache + k_base0 + (n_tile * 16) * kOfficialDidx + k_tile * 16;
                    nvcuda::wmma::load_matrix_sync(b0_frag, b0_ptr, kOfficialDidx);
                }
                if (page1_valid) {
                    const __nv_bfloat16* b1_ptr = k_idx_cache + k_base1 + (n_tile * 16) * kOfficialDidx + k_tile * 16;
                    nvcuda::wmma::load_matrix_sync(b1_frag, b1_ptr, kOfficialDidx);
                }

#define DO_GROUP32_PAIR_MMA(MTILE, C0, C1) \
                do { \
                    const __nv_bfloat16* a_ptr = q_shared + (MTILE) * 16 * kOfficialDidx + k_tile * 16; \
                    nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, kOfficialDidx); \
                    if (page0_valid) nvcuda::wmma::mma_sync((C0), a_frag, b0_frag, (C0)); \
                    if (page1_valid) nvcuda::wmma::mma_sync((C1), a_frag, b1_frag, (C1)); \
                } while (0)
                DO_GROUP32_PAIR_MMA(0, c00_frag, c10_frag);
                DO_GROUP32_PAIR_MMA(1, c01_frag, c11_frag);
                DO_GROUP32_PAIR_MMA(2, c02_frag, c12_frag);
                DO_GROUP32_PAIR_MMA(3, c03_frag, c13_frag);
#undef DO_GROUP32_PAIR_MMA
            }

            const int group = lane >> 2;
            const int quad = lane & 3;

#define ACCUM_GROUP32_PAIR_LOCAL(FRAG, MTILE, A0, A1, A2, A3) \
            do { \
                const float w0 = w_shared[(MTILE) * 16 + group]; \
                const float w1 = w_shared[(MTILE) * 16 + group + 8]; \
                float p0 = fmaxf((FRAG).x[0], 0.0f) * w0; \
                float p1 = fmaxf((FRAG).x[1], 0.0f) * w0; \
                p0 = fmaf(fmaxf((FRAG).x[2], 0.0f), w1, p0); \
                p1 = fmaf(fmaxf((FRAG).x[3], 0.0f), w1, p1); \
                float p2 = fmaxf((FRAG).x[4], 0.0f) * w0; \
                float p3 = fmaxf((FRAG).x[5], 0.0f) * w0; \
                p2 = fmaf(fmaxf((FRAG).x[6], 0.0f), w1, p2); \
                p3 = fmaf(fmaxf((FRAG).x[7], 0.0f), w1, p3); \
                (A0) += p0; \
                (A1) += p1; \
                (A2) += p2; \
                (A3) += p3; \
            } while (0)

            if (page0_valid) {
                float acc00 = 0.0f;
                float acc01 = 0.0f;
                float acc02 = 0.0f;
                float acc03 = 0.0f;
                ACCUM_GROUP32_PAIR_LOCAL(c00_frag, 0, acc00, acc01, acc02, acc03);
                ACCUM_GROUP32_PAIR_LOCAL(c01_frag, 1, acc00, acc01, acc02, acc03);
                ACCUM_GROUP32_PAIR_LOCAL(c02_frag, 2, acc00, acc01, acc02, acc03);
                ACCUM_GROUP32_PAIR_LOCAL(c03_frag, 3, acc00, acc01, acc02, acc03);
                acc00 = quad_group_reduce_sum(acc00);
                acc01 = quad_group_reduce_sum(acc01);
                acc02 = quad_group_reduce_sum(acc02);
                acc03 = quad_group_reduce_sum(acc03);
                if (group == 0) {
                    const int token0 = n_tile * 16 + quad * 2;
                    const int token1 = token0 + 1;
                    const int token2 = token0 + 8;
                    const int token3 = token0 + 9;
                    __nv_bfloat16* out_ptr = scores + b * max_seq_len + page_start0;
                    if (full_group) {
                        store_bf16_pair(out_ptr + token0, acc00, acc01);
                        store_bf16_pair(out_ptr + token2, acc02, acc03);
                    } else {
                        if (token1 < valid_count0) {
                            store_bf16_pair(out_ptr + token0, acc00, acc01);
                        } else if (token0 < valid_count0) {
                            out_ptr[token0] = __float2bfloat16(acc00);
                        }
                        if (token3 < valid_count0) {
                            store_bf16_pair(out_ptr + token2, acc02, acc03);
                        } else if (token2 < valid_count0) {
                            out_ptr[token2] = __float2bfloat16(acc02);
                        }
                    }
                }
            }

            if (page1_valid) {
                float acc10 = 0.0f;
                float acc11 = 0.0f;
                float acc12 = 0.0f;
                float acc13 = 0.0f;
                ACCUM_GROUP32_PAIR_LOCAL(c10_frag, 0, acc10, acc11, acc12, acc13);
                ACCUM_GROUP32_PAIR_LOCAL(c11_frag, 1, acc10, acc11, acc12, acc13);
                ACCUM_GROUP32_PAIR_LOCAL(c12_frag, 2, acc10, acc11, acc12, acc13);
                ACCUM_GROUP32_PAIR_LOCAL(c13_frag, 3, acc10, acc11, acc12, acc13);
                acc10 = quad_group_reduce_sum(acc10);
                acc11 = quad_group_reduce_sum(acc11);
                acc12 = quad_group_reduce_sum(acc12);
                acc13 = quad_group_reduce_sum(acc13);
                if (group == 0) {
                    const int token0 = n_tile * 16 + quad * 2;
                    const int token1 = token0 + 1;
                    const int token2 = token0 + 8;
                    const int token3 = token0 + 9;
                    __nv_bfloat16* out_ptr = scores + b * max_seq_len + page_start1;
                    if (full_group) {
                        store_bf16_pair(out_ptr + token0, acc10, acc11);
                        store_bf16_pair(out_ptr + token2, acc12, acc13);
                    } else {
                        if (token1 < valid_count1) {
                            store_bf16_pair(out_ptr + token0, acc10, acc11);
                        } else if (token0 < valid_count1) {
                            out_ptr[token0] = __float2bfloat16(acc10);
                        }
                        if (token3 < valid_count1) {
                            store_bf16_pair(out_ptr + token2, acc12, acc13);
                        } else if (token2 < valid_count1) {
                            out_ptr[token2] = __float2bfloat16(acc12);
                        }
                    }
                }
            }
#undef ACCUM_GROUP32_PAIR_LOCAL
        }
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
        if (MaxPages >= 256) {
            dim3 group_grid((unsigned)((MaxPages + 31) >> 5), (unsigned)B);
            dsa_indexer_scores_page64_group32_pair_kernel<<<group_grid, kPage64ThreadsPerBlock>>>(
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

        if (MaxPages >= 128) {
            dim3 group_grid((unsigned)((MaxPages + 7) >> 3), (unsigned)B);
            dsa_indexer_scores_page64_group8_pair_kernel<<<group_grid, kPage64ThreadsPerBlock>>>(
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

        if (MaxPages >= 64) {
            dim3 group_grid((unsigned)((MaxPages + 3) >> 2), (unsigned)B);
            dsa_indexer_scores_page64_group4_parallel_kernel<<<group_grid, kPage64ThreadsPerBlock>>>(
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
