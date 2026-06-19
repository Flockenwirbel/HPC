import tilelang
from tilelang import language as T


OFFICIAL_HIDX = 64
OFFICIAL_DIDX = 128
OFFICIAL_PAGE_SIZE = 64


@tilelang.jit
def _dsa_indexer_scores_empty_feature_tilelang_kernel(
    context_lens,
    scores,
    threads: int = 256,
):
    batch, max_seq_len = T.const("batch, max_seq_len")
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    context_lens: T.Tensor[[batch], index_dtype]
    scores: T.Tensor[[batch, max_seq_len], dtype]

    with T.Kernel(max_seq_len, batch, threads=threads) as (s, b):
        if s < context_lens[b]:
            scores[b, s] = 0.0
        else:
            scores[b, s] = -T.infinity(accum_dtype)


@tilelang.jit
def _dsa_indexer_scores_generic_tilelang_kernel(
    q_idx,
    k_idx_cache,
    w_idx,
    block_table,
    context_lens,
    scores,
    HIDX: int,
    DIDX: int,
    PAGE_SIZE: int,
    threads: int = 1,
):
    batch, max_pages, max_seq_len, cache_tokens = T.const(
        "batch, max_pages, max_seq_len, cache_tokens"
    )

    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    q_idx: T.Tensor[[batch * HIDX, DIDX], dtype]
    k_idx_cache: T.Tensor[[cache_tokens, DIDX], dtype]
    w_idx: T.Tensor[[batch, HIDX], dtype]
    block_table: T.Tensor[[batch, max_pages], index_dtype]
    context_lens: T.Tensor[[batch], index_dtype]
    scores: T.Tensor[[batch, max_seq_len], dtype]

    with T.Kernel(max_seq_len, batch, threads=threads) as (s, b):
        visible = context_lens[b]

        if s < visible:
            logical_page = s // PAGE_SIZE
            page_offset = s - logical_page * PAGE_SIZE
            physical_page = block_table[b, logical_page]
            acc = T.alloc_var(accum_dtype)
            acc = 0.0

            for h in T.serial(HIDX):
                dot = T.alloc_var(accum_dtype)
                dot = 0.0
                for d in T.serial(DIDX):
                    dot += q_idx[b * HIDX + h, d] * k_idx_cache[physical_page * PAGE_SIZE + page_offset, d]
                if dot > 0.0:
                    acc += dot * w_idx[b, h]

            scores[b, s] = acc
        else:
            scores[b, s] = -T.infinity(accum_dtype)


@tilelang.jit
def _dsa_indexer_scores_bf16_tilelang_kernel(
    q_idx,
    k_idx_cache,
    w_idx,
    block_table,
    context_lens,
    scores,
    HIDX: int = 64,
    DIDX: int = 128,
    PAGE_SIZE: int = 64,
    GROUP_PAGES: int = 1,
    threads: int = 512,
):
    batch, max_pages, max_seq_len, cache_tokens = T.const(
        "batch, max_pages, max_seq_len, cache_tokens"
    )

    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    q_idx: T.Tensor[[batch * HIDX, DIDX], dtype]
    k_idx_cache: T.Tensor[[cache_tokens, DIDX], dtype]
    w_idx: T.Tensor[[batch, HIDX], dtype]
    block_table: T.Tensor[[batch, max_pages], index_dtype]
    context_lens: T.Tensor[[batch], index_dtype]
    scores: T.Tensor[[batch, max_seq_len], dtype]

    with T.Kernel(T.ceildiv(max_pages, GROUP_PAGES), batch, threads=threads) as (pid_group, b):
        q_shared = T.alloc_shared([HIDX, DIDX], dtype)
        k_shared = T.alloc_shared([PAGE_SIZE, DIDX], dtype)
        dots = T.alloc_fragment([PAGE_SIZE, HIDX], accum_dtype)
        weighted = T.alloc_fragment([PAGE_SIZE, HIDX], accum_dtype)
        weights = T.alloc_fragment([HIDX], accum_dtype)
        out = T.alloc_fragment([PAGE_SIZE], accum_dtype)

        first_logical_page = pid_group * GROUP_PAGES
        first_page_start = first_logical_page * PAGE_SIZE
        visible = context_lens[b]

        if first_logical_page < max_pages:
            if first_page_start < visible:
                T.copy(q_idx[b * HIDX, 0], q_shared)

                for h in T.Parallel(HIDX):
                    weights[h] = w_idx[b, h]

                for page_i in T.serial(GROUP_PAGES):
                    logical_page = first_logical_page + page_i
                    page_start = logical_page * PAGE_SIZE

                    if logical_page < max_pages:
                        if page_start < visible:
                            physical_page = block_table[b, logical_page]
                            T.copy(k_idx_cache[physical_page * PAGE_SIZE, 0], k_shared)

                            T.gemm(
                                k_shared,
                                q_shared,
                                dots,
                                transpose_B=True,
                                clear_accum=True,
                                policy=T.GemmWarpPolicy.FullCol,
                            )

                            for n, h in T.Parallel(PAGE_SIZE, HIDX):
                                weighted[n, h] = T.max(dots[n, h], 0.0) * weights[h]

                            T.reduce_sum(weighted, out, dim=-1, clear=True)

                            for n in T.Parallel(PAGE_SIZE):
                                token = page_start + n
                                if token < visible:
                                    scores[b, token] = out[n]
                                else:
                                    scores[b, token] = -T.infinity(accum_dtype)
                        else:
                            for n in T.Parallel(PAGE_SIZE):
                                scores[b, page_start + n] = -T.infinity(accum_dtype)
            else:
                for page_i in T.serial(GROUP_PAGES):
                    logical_page = first_logical_page + page_i
                    page_start = logical_page * PAGE_SIZE
                    if logical_page < max_pages:
                        for n in T.Parallel(PAGE_SIZE):
                            scores[b, page_start + n] = -T.infinity(accum_dtype)


def _group_pages(max_pages):
    if max_pages >= 256:
        return 8
    if max_pages >= 64:
        return 4
    return 1


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

    if Hidx <= 0 or Didx <= 0:
        _dsa_indexer_scores_empty_feature_tilelang_kernel(
            context_lens.view(B),
            scores.view(B, max_seq_len),
            threads=256,
        )
        return

    if Hidx == OFFICIAL_HIDX and Didx == OFFICIAL_DIDX and PageSize == OFFICIAL_PAGE_SIZE:
        _dsa_indexer_scores_bf16_tilelang_kernel(
            q_idx.view(B * Hidx, Didx),
            k_idx_cache.view(-1, Didx),
            w_idx.view(B, Hidx),
            block_table.view(B, MaxPages),
            context_lens.view(B),
            scores.view(B, max_seq_len),
            HIDX=64,
            DIDX=128,
            PAGE_SIZE=64,
            GROUP_PAGES=_group_pages(MaxPages),
            threads=512,
        )
        return

    _dsa_indexer_scores_generic_tilelang_kernel(
        q_idx.view(B * Hidx, Didx),
        k_idx_cache.view(-1, Didx),
        w_idx.view(B, Hidx),
        block_table.view(B, MaxPages),
        context_lens.view(B),
        scores.view(B, max_seq_len),
        HIDX=Hidx,
        DIDX=Didx,
        PAGE_SIZE=PageSize,
        threads=1,
    )
