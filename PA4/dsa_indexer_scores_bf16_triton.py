import triton
import triton.language as tl


OFFICIAL_HIDX = 64
OFFICIAL_DIDX = 128
OFFICIAL_PAGE_SIZE = 64
SMALL_TOTAL_PAGES_THRESHOLD = 256


@triton.jit
def _dsa_indexer_scores_page_full_kernel(
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
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
    ],
    key=["total_pages", "max_pages"],
)
@triton.jit
def _dsa_indexer_scores_page_split_kernel(
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
    BLOCK_N: tl.constexpr,
):
    pid_page = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)

    b = pid_page // max_pages
    logical_page = pid_page - b * max_pages
    page_start = logical_page * PAGE_SIZE

    offs_h = tl.arange(0, HIDX)
    offs_n = pid_block * BLOCK_N + tl.arange(0, BLOCK_N)

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
        offsets=(pid_block * BLOCK_N, 0),
        block_shape=(BLOCK_N, DIDX),
        order=(1, 0),
    )
    k = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zero")

    dots = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    dots = tl.maximum(dots, 0.0)
    scores_block = tl.sum(dots * w[:, None], axis=0)
    out = tl.where(token_valid, scores_block, float("-inf"))

    score_ptrs = scores_ptr + b * max_seq_len + token_global
    tl.store(score_ptrs, out.to(tl.bfloat16), mask=offs_n < PAGE_SIZE)


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
    if Hidx != OFFICIAL_HIDX or Didx != OFFICIAL_DIDX or PageSize != OFFICIAL_PAGE_SIZE:
        raise NotImplementedError(
            "Experimental Triton version only supports Hidx=64, Didx=128, PageSize=64"
        )

    max_seq_len = MaxPages * PageSize
    if B <= 0 or MaxPages <= 0 or max_seq_len <= 0:
        return

    total_pages = B * MaxPages

    if total_pages <= SMALL_TOTAL_PAGES_THRESHOLD:
        grid = (total_pages,)
        _dsa_indexer_scores_page_full_kernel[grid](
            q_idx,
            k_idx_cache,
            w_idx,
            block_table,
            context_lens,
            scores,
            MaxPages,
            max_seq_len,
            HIDX=64,
            DIDX=128,
            PAGE_SIZE=64,
            num_warps=8,
            num_stages=2,
        )
        return

    grid = lambda meta: (total_pages, triton.cdiv(OFFICIAL_PAGE_SIZE, meta["BLOCK_N"]))
    _dsa_indexer_scores_page_split_kernel[grid](
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
