import triton
import triton.language as tl


OFFICIAL_HIDX = 64
OFFICIAL_DIDX = 128
OFFICIAL_PAGE_SIZE = 64
SMALL_TOTAL_PAGES_THRESHOLD = 2048


@triton.jit
def _dsa_indexer_scores_empty_feature_kernel(
    context_lens_ptr,
    scores_ptr,
    max_seq_len,
):
    pid = tl.program_id(axis=0)
    b = pid // max_seq_len
    s = pid - b * max_seq_len

    visible = tl.load(context_lens_ptr + b)
    out = tl.where(s < visible, 0.0, float("-inf"))
    tl.store(scores_ptr + b * max_seq_len + s, out.to(tl.bfloat16))


@triton.jit
def _dsa_indexer_scores_generic_kernel(
    q_idx_ptr,
    k_idx_cache_ptr,
    w_idx_ptr,
    block_table_ptr,
    context_lens_ptr,
    scores_ptr,
    max_pages,
    max_seq_len,
    HIDX: tl.constexpr,
    DIDX: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // max_seq_len
    s = pid - b * max_seq_len

    visible = tl.load(context_lens_ptr + b)
    out_ptr = scores_ptr + b * max_seq_len + s

    if s < visible:
        logical_page = s // PAGE_SIZE
        page_offset = s - logical_page * PAGE_SIZE
        physical_page = tl.load(block_table_ptr + b * max_pages + logical_page)

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        token_k_base = (physical_page * PAGE_SIZE + page_offset) * DIDX
        acc = tl.full((), 0.0, tl.float32)

        for h_base in tl.range(0, HIDX, BLOCK_H, loop_unroll_factor=1):
            h = h_base + offs_h
            h_mask = h < HIDX
            dots = tl.full((BLOCK_H,), 0.0, tl.float32)

            for d_base in tl.range(0, DIDX, BLOCK_D, loop_unroll_factor=1):
                d = d_base + offs_d
                d_mask = d < DIDX
                q = tl.load(
                    q_idx_ptr + ((b * HIDX + h[:, None]) * DIDX + d[None, :]),
                    mask=h_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                k = tl.load(
                    k_idx_cache_ptr + token_k_base + d,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)
                dots += tl.sum(q * k[None, :], axis=1)

            w = tl.load(w_idx_ptr + b * HIDX + h, mask=h_mask, other=0.0).to(tl.float32)
            acc += tl.sum(tl.maximum(dots, 0.0) * w, axis=0)

        tl.store(out_ptr, acc.to(tl.bfloat16))
    else:
        out = tl.full((), float("-inf"), tl.float32)
        tl.store(out_ptr, out.to(tl.bfloat16))


@triton.jit
def _dsa_indexer_scores_small_full_kernel(
    q_idx_ptr,
    k_idx_cache_ptr,
    w_idx_ptr,
    block_table_ptr,
    context_lens_ptr,
    scores_ptr,
    total_pages,
    max_pages,
    max_seq_len,
    HIDX: tl.constexpr,
    DIDX: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    pid_page = tl.program_id(axis=0)

    b = pid_page // max_pages
    logical_page = pid_page - b * max_pages
    page_start = logical_page * PAGE_SIZE

    offs_h = tl.arange(0, HIDX)
    offs_d = tl.arange(0, DIDX)
    offs_n = tl.arange(0, PAGE_SIZE)

    visible = tl.load(context_lens_ptr + b)
    token_global = page_start + offs_n
    token_valid = token_global < visible
    page_valid = page_start < visible

    physical_page = tl.load(
        block_table_ptr + b * max_pages + logical_page,
        mask=page_valid,
        other=0,
    )

    q_ptrs = q_idx_ptr + ((b * HIDX + offs_h[:, None]) * DIDX + offs_d[None, :])
    q = tl.load(q_ptrs)

    w = tl.load(w_idx_ptr + b * HIDX + offs_h).to(tl.float32)

    k_ptrs = k_idx_cache_ptr + ((physical_page * PAGE_SIZE + offs_n[:, None]) * DIDX + offs_d[None, :])
    k = tl.load(k_ptrs, mask=token_valid[:, None], other=0)

    dots = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    dots = tl.maximum(dots, 0.0)
    scores_block = tl.sum(dots * w[:, None], axis=0)
    out = tl.where(token_valid, scores_block, float("-inf"))

    score_ptrs = scores_ptr + b * max_seq_len + token_global
    tl.store(score_ptrs, out.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=["total_pages", "max_pages"],
)
@triton.jit
def _dsa_indexer_scores_large_full_kernel(
    q_idx_ptr,
    k_idx_cache_ptr,
    w_idx_ptr,
    block_table_ptr,
    context_lens_ptr,
    scores_ptr,
    total_pages,
    max_pages,
    max_seq_len,
    HIDX: tl.constexpr,
    DIDX: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    pid_page = tl.program_id(axis=0)

    b = pid_page // max_pages
    logical_page = pid_page - b * max_pages
    page_start = logical_page * PAGE_SIZE

    offs_h = tl.arange(0, HIDX)
    offs_n = tl.arange(0, PAGE_SIZE)

    visible = tl.load(context_lens_ptr + b)
    token_global = page_start + offs_n
    token_valid = token_global < visible
    page_valid = page_start < visible

    physical_page = tl.load(
        block_table_ptr + b * max_pages + logical_page,
        mask=page_valid,
        other=0,
    )

    q_block_ptr = tl.make_block_ptr(
        base=q_idx_ptr + b * HIDX * DIDX,
        shape=(HIDX, DIDX),
        strides=(DIDX, 1),
        offsets=(0, 0),
        block_shape=(HIDX, DIDX),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr)

    w = tl.load(w_idx_ptr + b * HIDX + offs_h).to(tl.float32)

    k_block_ptr = tl.make_block_ptr(
        base=k_idx_cache_ptr + physical_page * PAGE_SIZE * DIDX,
        shape=(PAGE_SIZE, DIDX),
        strides=(DIDX, 1),
        offsets=(0, 0),
        block_shape=(PAGE_SIZE, DIDX),
        order=(1, 0),
    )
    k = tl.load(k_block_ptr)

    dots = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    dots = tl.maximum(dots, 0.0)
    scores_block = tl.sum(dots * w[:, None], axis=0)
    out = tl.where(token_valid, scores_block, float("-inf"))

    score_ptrs = scores_ptr + b * max_seq_len + token_global
    tl.store(score_ptrs, out.to(tl.bfloat16))


def run_kernel(
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
    PageSize,
):
    max_seq_len = MaxPages * PageSize
    if B <= 0 or MaxPages <= 0 or PageSize <= 0 or max_seq_len <= 0:
        return

    total_pages = B * MaxPages

    if Hidx <= 0 or Didx <= 0:
        _dsa_indexer_scores_empty_feature_kernel[(B * max_seq_len,)](
            context_lens,
            scores,
            max_seq_len,
            num_warps=4,
        )
        return

    if Hidx != OFFICIAL_HIDX or Didx != OFFICIAL_DIDX or PageSize != OFFICIAL_PAGE_SIZE:
        block_h = min(32, triton.next_power_of_2(Hidx))
        block_d = min(64, triton.next_power_of_2(Didx))
        _dsa_indexer_scores_generic_kernel[(B * max_seq_len,)](
            q_idx,
            k_idx_cache,
            w_idx,
            block_table,
            context_lens,
            scores,
            MaxPages,
            max_seq_len,
            HIDX=Hidx,
            DIDX=Didx,
            PAGE_SIZE=PageSize,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
        return

    grid = (total_pages,)

    if total_pages <= SMALL_TOTAL_PAGES_THRESHOLD:
        _dsa_indexer_scores_small_full_kernel[grid](
            q_idx,
            k_idx_cache,
            w_idx,
            block_table,
            context_lens,
            scores,
            total_pages,
            MaxPages,
            max_seq_len,
            HIDX=64,
            DIDX=128,
            PAGE_SIZE=64,
            num_warps=8,
            num_stages=2,
        )
        return

    _dsa_indexer_scores_large_full_kernel[grid](
        q_idx,
        k_idx_cache,
        w_idx,
        block_table,
        context_lens,
        scores,
        total_pages,
        MaxPages,
        max_seq_len,
        HIDX=64,
        DIDX=128,
        PAGE_SIZE=64,
    )
