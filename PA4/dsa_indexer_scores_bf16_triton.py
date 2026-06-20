import triton
import triton.language as tl


@triton.jit
def _dsa_empty_feature_kernel(
    context_lens,
    scores,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_s = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    visible = tl.load(context_lens + b)
    vals = tl.where(offs < visible, 0.0, -float("inf"))
    tl.store(scores + b * MAX_SEQ_LEN + offs, vals, mask=offs < MAX_SEQ_LEN)


@triton.jit
def _dsa_indexer_scores_official_2page_kernel(
    q_idx,
    k_idx_cache,
    w_idx,
    block_table,
    context_lens,
    scores,
    MAX_PAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    b = tl.program_id(1)

    max_seq_len = MAX_PAGES * 64
    tile_start = tile_id * 128
    visible = tl.load(context_lens + b, eviction_policy="evict_last")

    offs_n = tl.arange(0, 128)
    offs_h = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)
    tokens = tile_start + offs_n
    neg_inf = tl.full((128,), -float("inf"), tl.float32)

    if tile_start < visible:
        q = tl.load(
            q_idx + ((b * 64 + offs_h[:, None]) * 128 + offs_d[None, :]),
            eviction_policy="evict_last",
        )
        weights = tl.load(
            w_idx + b * 64 + offs_h,
            eviction_policy="evict_last",
        ).to(tl.float32)

        # Full hot path for official large cases.  A 128-token tile is exactly
        # two logical pages; load the two physical page ids once and keep both K
        # page loads affine/contiguous instead of doing per-token block_table
        # gathers.  The fallback below handles the one partial context/tail tile.
        if (tile_start + 128 <= visible) & (tile_start + 128 <= max_seq_len):
            logical_page0 = tile_start // 64
            physical_page0 = tl.load(
                block_table + b * MAX_PAGES + logical_page0,
                eviction_policy="evict_first",
            )
            physical_page1 = tl.load(
                block_table + b * MAX_PAGES + logical_page0 + 1,
                eviction_policy="evict_first",
            )

            physical_pages = tl.where(offs_n < 64, physical_page0, physical_page1)
            page_offsets = offs_n - (offs_n // 64) * 64
            k = tl.load(
                k_idx_cache
                + ((physical_pages[:, None] * 64 + page_offsets[:, None]) * 128 + offs_d[None, :]),
                eviction_policy="evict_first",
            )

            # Produce [token, head] so the epilogue reduces the last axis
            # (heads) instead of reducing across the first dimension of a
            # [head, token] accumulator.
            dots = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
            score = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], axis=1)
            tl.store(scores + b * max_seq_len + tokens, score)
        else:
            in_seq = tokens < max_seq_len
            token_valid = tokens < visible
            logical_pages = tokens // 64
            page_offsets = tokens - logical_pages * 64
            load_mask = token_valid & in_seq
            physical_pages = tl.load(
                block_table + b * MAX_PAGES + logical_pages,
                mask=load_mask,
                other=0,
                eviction_policy="evict_first",
            )
            k = tl.load(
                k_idx_cache
                + ((physical_pages[:, None] * 64 + page_offsets[:, None]) * 128 + offs_d[None, :]),
                mask=load_mask[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            dots = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
            score = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], axis=1)
            tl.store(
                scores + b * max_seq_len + tokens,
                tl.where(token_valid, score, neg_inf),
                mask=in_seq,
            )
    else:
        in_seq = tokens < max_seq_len
        tl.store(scores + b * max_seq_len + tokens, neg_inf, mask=in_seq)


@triton.jit
def _dsa_indexer_scores_official_group_page_kernel(
    q_idx,
    k_idx_cache,
    w_idx,
    block_table,
    context_lens,
    scores,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
):
    group_id = tl.program_id(0)
    b = tl.program_id(1)

    max_seq_len = MAX_PAGES * 64
    first_page = group_id * GROUP_PAGES
    first_start = first_page * 64
    visible = tl.load(context_lens + b, eviction_policy="evict_last")

    offs_n = tl.arange(0, 64)
    offs_h = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)
    neg_inf = tl.full((64,), -float("inf"), tl.float32)

    if first_start < visible:
        q = tl.load(
            q_idx + ((b * 64 + offs_h[:, None]) * 128 + offs_d[None, :]),
            eviction_policy="evict_last",
        )
        weights = tl.load(
            w_idx + b * 64 + offs_h,
            eviction_policy="evict_last",
        ).to(tl.float32)

        full_group = (first_page + GROUP_PAGES <= MAX_PAGES) & (first_start + GROUP_PAGES * 64 <= visible)
        if full_group:
            for page_i in tl.static_range(0, GROUP_PAGES):
                logical_page = first_page + page_i
                page_start = first_start + page_i * 64
                physical_page = tl.load(
                    block_table + b * MAX_PAGES + logical_page,
                    eviction_policy="evict_first",
                )
                k = tl.load(
                    k_idx_cache + ((physical_page * 64 + offs_n[:, None]) * 128 + offs_d[None, :]),
                    eviction_policy="evict_first",
                )
                dots = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
                score = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], axis=1)
                tl.store(scores + b * max_seq_len + page_start + offs_n, score)
        else:
            for page_i in tl.static_range(0, GROUP_PAGES):
                logical_page = first_page + page_i
                page_start = first_start + page_i * 64
                page_in_range = logical_page < MAX_PAGES
                page_has_visible = page_in_range & (page_start < visible)
                out_ptrs = scores + b * max_seq_len + page_start + offs_n

                if page_has_visible:
                    physical_page = tl.load(
                        block_table + b * MAX_PAGES + logical_page,
                        eviction_policy="evict_first",
                    )
                    k = tl.load(
                        k_idx_cache + ((physical_page * 64 + offs_n[:, None]) * 128 + offs_d[None, :]),
                        eviction_policy="evict_first",
                    )
                    dots = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
                    score = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], axis=1)
                    if page_start + 64 <= visible:
                        tl.store(out_ptrs, score)
                    else:
                        token_valid = page_start + offs_n < visible
                        tl.store(out_ptrs, tl.where(token_valid, score, neg_inf))
                else:
                    tl.store(out_ptrs, neg_inf, mask=page_in_range)
    else:
        for page_i in tl.static_range(0, GROUP_PAGES):
            logical_page = first_page + page_i
            page_start = first_start + page_i * 64
            page_in_range = logical_page < MAX_PAGES
            tl.store(
                scores + b * max_seq_len + page_start + offs_n,
                neg_inf,
                mask=page_in_range,
            )


@triton.jit
def _dsa_indexer_scores_kernel(
    q_idx,
    k_idx_cache,
    w_idx,
    block_table,
    context_lens,
    scores,
    HIDX: tl.constexpr,
    DIDX: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_id = tl.program_id(0)
    b = tl.program_id(1)

    max_seq_len = MAX_PAGES * PAGE_SIZE
    tile_start = tile_id * BLOCK_N
    visible = tl.load(context_lens + b, eviction_policy="evict_last")

    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    tokens = tile_start + offs_n
    in_seq = tokens < max_seq_len
    token_valid = tokens < visible
    neg_inf = tl.full((BLOCK_N,), -float("inf"), tl.float32)

    if tile_start < visible:
        q = tl.load(
            q_idx + ((b * HIDX + offs_h[:, None]) * DIDX + offs_d[None, :]),
            mask=(offs_h[:, None] < HIDX) & (offs_d[None, :] < DIDX),
            other=0.0,
            eviction_policy="evict_last",
        )
        weights = tl.load(
            w_idx + b * HIDX + offs_h,
            mask=offs_h < HIDX,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        logical_pages = tokens // PAGE_SIZE
        page_offsets = tokens - logical_pages * PAGE_SIZE
        valid_load_tokens = token_valid & in_seq
        physical_pages = tl.load(
            block_table + b * MAX_PAGES + logical_pages,
            mask=valid_load_tokens,
            other=0,
            eviction_policy="evict_first",
        )
        k = tl.load(
            k_idx_cache
            + ((physical_pages[:, None] * PAGE_SIZE + page_offsets[:, None]) * DIDX + offs_d[None, :]),
            mask=valid_load_tokens[:, None] & (offs_d[None, :] < DIDX),
            other=0.0,
            eviction_policy="evict_first",
        )

        dots = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
        score = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], axis=1)
        tl.store(
            scores + b * max_seq_len + tokens,
            tl.where(token_valid, score, neg_inf),
            mask=in_seq,
        )
    else:
        tl.store(scores + b * max_seq_len + tokens, neg_inf, mask=in_seq)


def _next_power_of_2(x):
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _choose_block_n(B, max_pages, page_size):
    total_pages = int(B) * int(max_pages)
    if page_size == 64 and total_pages >= 512:
        return 128
    return max(16, _next_power_of_2(page_size))


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
    B = int(B)
    Hidx = int(Hidx)
    Didx = int(Didx)
    MaxPages = int(MaxPages)
    PageSize = int(PageSize)
    max_seq_len = MaxPages * PageSize

    if B <= 0 or MaxPages <= 0 or PageSize <= 0 or max_seq_len <= 0:
        return

    if Hidx <= 0 or Didx <= 0:
        block = 256
        grid = (triton.cdiv(max_seq_len, block), B)
        _dsa_empty_feature_kernel[grid](
            context_lens,
            scores,
            MAX_SEQ_LEN=max_seq_len,
            BLOCK_SIZE=block,
            num_warps=8,
        )
        return

    total_pages = B * MaxPages
    if Hidx == 64 and Didx == 128 and PageSize == 64 and total_pages >= 4096:
        # Group4 was the sweet spot in benchmarking: group8 reduced parallelism
        # and increased the unrolled program body enough to regress case7-10.
        group_pages = 4
        grid = (triton.cdiv(MaxPages, group_pages), B)
        _dsa_indexer_scores_official_group_page_kernel[grid](
            q_idx,
            k_idx_cache,
            w_idx,
            block_table,
            context_lens,
            scores,
            MAX_PAGES=MaxPages,
            GROUP_PAGES=group_pages,
            num_warps=4,
            num_stages=3,
        )
        return

    if Hidx == 64 and Didx == 128 and PageSize == 64 and total_pages >= 512:
        grid = (triton.cdiv(max_seq_len, 128), B)
        _dsa_indexer_scores_official_2page_kernel[grid](
            q_idx,
            k_idx_cache,
            w_idx,
            block_table,
            context_lens,
            scores,
            MAX_PAGES=MaxPages,
            num_warps=8,
            num_stages=3,
        )
        return

    block_n = _choose_block_n(B, MaxPages, PageSize)
    block_h = max(16, _next_power_of_2(Hidx))
    block_d = max(32, _next_power_of_2(Didx))

    grid = (triton.cdiv(max_seq_len, block_n), B)
    _dsa_indexer_scores_kernel[grid](
        q_idx,
        k_idx_cache,
        w_idx,
        block_table,
        context_lens,
        scores,
        HIDX=Hidx,
        DIDX=Didx,
        MAX_PAGES=MaxPages,
        PAGE_SIZE=PageSize,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=3,
    )
