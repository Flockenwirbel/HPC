#include "apsp.h"

constexpr int INF = 0x3f3f3f3f;

namespace {

// Phase 1: Update pivot block
template <int B, int R>
__global__ void phase1(int n, int k_start, int *graph) {
    const int BS = B * R;
    // global pos
    int g_i = k_start + threadIdx.y * R;
    int g_j = k_start + threadIdx.x * R;

    // reg
    int reg[R][R];
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            int idx = (g_i + i) * n + (g_j + j);
            reg[i][j] = (g_i + i < n && g_j + j < n) ? graph[idx] : INF;
        }
    }

    // shared mem
    __shared__ int shared[BS][BS];
    for (int ri = 0; ri < R; ++ri) {
        for (int rj = 0; rj < R; ++rj) {
            int posx = threadIdx.x * R + rj;
            int posy = threadIdx.y * R + ri;
            if (posx < BS && posy < BS) shared[posy][posx] = reg[ri][rj];
        }
    }

    __syncthreads();

    for (int k = 0; k < BS; ++k) {
        for (int ri = 0; ri < R; ++ri) {
            for (int rj = 0; rj < R; ++rj) {
                int posx = threadIdx.x * R + rj;
                int posy = threadIdx.y * R + ri;
                reg[ri][rj] = min(reg[ri][rj], shared[posy][k] + shared[k][posx]);
            }
        }

        // write back to shared mem
        for (int ri = 0; ri < R; ++ri) {
            for (int rj = 0; rj < R; ++rj) {
                int posx = threadIdx.x * R + rj;
                int posy = threadIdx.y * R + ri;
                shared[posy][posx] = reg[ri][rj];
            }
        }

        __syncthreads();
    }

    // write back to global mem
    for (int ri = 0; ri < R; ++ri) {
        for (int rj = 0; rj < R; ++rj) {
            int idx = (g_i + ri) * n + (g_j + rj);
            if (g_i + ri < n && g_j + rj < n) graph[idx] = reg[ri][rj];
        }
    }
}

template <int B, int R>
__global__ void phase2_row(int n, int kb, int *graph) {
    const int BS = B * R;
    int g_j = blockIdx.x * BS;
    int g_i = kb * BS;
    int l_i = threadIdx.y * R;
    int l_j = threadIdx.x * R;

    __shared__ int shared_pivot[BS][BS];
    __shared__ int shared_cur[BS][BS];

    // read from pivot to shared pivot
    int pivot_i = kb * BS + l_i;
    int pivot_j = kb * BS + l_j;
    for (int ri = 0; ri < R; ++ri) {
        for (int rj = 0; rj < R; ++rj) {
            int idx = (pivot_i + ri) * n + (pivot_j + rj);
            shared_pivot[threadIdx.y * R + ri][threadIdx.x * R + rj] = (pivot_i + ri < n && pivot_j + rj < n) ? graph[idx] : INF;
        }
    }

    // read from block(kb, j) to shared cur
    int cur_i = kb * BS + l_i;
    int cur_j = blockIdx.x * BS + l_j;
    for (int ri = 0; ri < R; ++ri) {
        for (int rj = 0; rj < R; ++rj) {
            int idx = (cur_i + ri) * n + (cur_j + rj);
            shared_cur[threadIdx.y * R + ri][threadIdx.x * R + rj] = (cur_i + ri < n && cur_j + rj < n) ? graph[idx] : INF;
        }
    }

    __syncthreads();

    // load reg
    int reg[R][R];
    for (int ri = 0; ri < R; ++ri) 
        for (int rj = 0; rj < R; ++rj)
            reg[ri][rj] = shared_cur[threadIdx.y * R + ri][threadIdx.x * R + rj];
    

    for (int k = 0; k < BS; ++k) {
        for (int ri = 0; ri < R; ++ri)
            for (int rj = 0; rj < R; ++rj) {
                int posx = threadIdx.x * R + rj;
                int posy = threadIdx.y * R + ri;
                reg[ri][rj] = min(reg[ri][rj], shared_pivot[posy][k] + shared_cur[k][posx]);
            }
        

        for (int ri = 0; ri < R; ++ri) 
            for (int rj = 0; rj < R; ++rj) {
                int posx = threadIdx.x * R + rj;
                int posy = threadIdx.y * R + ri;
                shared_cur[posy][posx] = reg[ri][rj];
            }

        __syncthreads();
    }

    // write back to global mem
    for (int ri = 0; ri < R; ++ri)
        for (int rj = 0; rj < R; ++rj) {
            int idx = (g_i + l_i + ri) * n + (g_j + l_j + rj);
            if (g_i + l_i + ri < n && g_j + l_j + rj < n) graph[idx] = reg[ri][rj];
        }

}

template <int B, int R>
__global__ void phase2_col(int n, int kb, int *graph) {
        const int BS = B * R;
    int g_j = kb * BS;
    int g_i = blockIdx.x * BS;
    int l_i = threadIdx.y * R;
    int l_j = threadIdx.x * R;

    __shared__ int shared_pivot[BS][BS];
    __shared__ int shared_cur[BS][BS];

    // read from pivot to shared pivot
    int pivot_i = kb * BS + l_i;
    int pivot_j = kb * BS + l_j;
    for (int ri = 0; ri < R; ++ri) {
        for (int rj = 0; rj < R; ++rj) {
            int idx = (pivot_i + ri) * n + (pivot_j + rj);
            shared_pivot[threadIdx.y * R + ri][threadIdx.x * R + rj] = (pivot_i + ri < n && pivot_j + rj < n) ? graph[idx] : INF;
        }
    }

    // read from block(kb, j) to shared cur
    int cur_i = blockIdx.x * BS + l_i;
    int cur_j = kb * BS + l_j;
    for (int ri = 0; ri < R; ++ri) {
        for (int rj = 0; rj < R; ++rj) {
            int idx = (cur_i + ri) * n + (cur_j + rj);
            int posx = threadIdx.x * R + rj;
            int posy = threadIdx.y * R + ri;
            shared_cur[posy][posx] = (cur_i + ri < n && cur_j + rj < n) ? graph[idx] : INF;
        }
    }

    __syncthreads();

    // load reg
    int reg[R][R];
    for (int ri = 0; ri < R; ++ri) 
        for (int rj = 0; rj < R; ++rj)
            reg[ri][rj] = shared_cur[threadIdx.y * R + ri][threadIdx.x * R + rj];
    

    for (int k = 0; k < BS; ++k) {
        for (int ri = 0; ri < R; ++ri)
            for (int rj = 0; rj < R; ++rj) {
                int posx = threadIdx.x * R + rj;
                int posy = threadIdx.y * R + ri;
                reg[ri][rj] = min(reg[ri][rj], shared_cur[posy][k] + shared_pivot[k][posx]);
            }
        

        for (int ri = 0; ri < R; ++ri) 
            for (int rj = 0; rj < R; ++rj) {
                int posx = threadIdx.x * R + rj;
                int posy = threadIdx.y * R + ri;
                shared_cur[posy][posx] = reg[ri][rj];
            }

        __syncthreads();
    }

    // write back to global mem
    for (int ri = 0; ri < R; ++ri)
        for (int rj = 0; rj < R; ++rj) {
            int idx = (g_i + l_i + ri) * n + (g_j + l_j + rj);
            if (g_i + l_i + ri < n && g_j + l_j + rj < n) graph[idx] = reg[ri][rj];
        }
}


template <int B, int R>
__global__ void phase3(int n, int kb, int *graph) {
    const int BS = B * R;
    int g_i = blockIdx.y * BS;
    int g_j = blockIdx.x * BS;
    if (blockIdx.y == kb || blockIdx.x == kb) return;

    int l_i = threadIdx.y * R;
    int l_j = threadIdx.x * R;

    __shared__ int shared_row[BS][BS];
    __shared__ int shared_col[BS][BS];

    // read pivot to shared row
    int row_i = kb * BS + l_i;
    int row_j = g_j + l_j;
    for (int ri = 0; ri < R; ++ri) 
        for (int rj = 0; rj < R; ++rj) 
            shared_row[l_i + ri][l_j + rj] =  (row_i + ri < n && row_j + rj < n) ? graph[(row_i + ri) * n + (row_j + rj)] : INF;

    // read pivot to shared col
    int col_i = g_i + l_i;
    int col_j = kb * BS + l_j;
    for (int ri = 0; ri < R; ++ri) 
        for (int rj = 0; rj < R; ++rj) 
            shared_col[l_i + ri][l_j + rj] = (col_i + ri < n && col_j + rj < n) ? graph[(col_i + ri) * n + (col_j + rj)] : INF;

    __syncthreads();

    // read current block to reg
    int reg[R][R];
    for (int ri = 0; ri < R; ++ri)
        for (int rj = 0; rj < R; ++rj)
            reg[ri][rj] = (g_i + l_i + ri < n && g_j + l_j + rj < n) ? graph[(g_i + l_i + ri) * n + (g_j + l_j + rj)] : INF;
    
    for (int k = 0; k < BS; ++k) 
        for (int ri = 0; ri < R; ++ri)
            for (int rj = 0; rj < R; ++rj) {
                int posx = threadIdx.x * R + rj;
                int posy = threadIdx.y * R + ri;
                reg[ri][rj] = min(reg[ri][rj], shared_col[posy][k] + shared_row[k][posx]);
            }
    
    // write back to global mem
    for (int ri = 0; ri < R; ++ri)
        for (int rj = 0; rj < R; ++rj) {
            int idx = (g_i + l_i + ri) * n + (g_j + l_j + rj);
            if (g_i + l_i + ri < n && g_j + l_j + rj < n) graph[idx] = reg[ri][rj];
        }
}
}

void apsp(int n, /* device */ int *graph) {
    const int B = 16;        // 16 threads per dim
    const int R = 4;         // 4 elements per thread per dim
    const int block_size = B * R;  // 64

    int num_blocks = (n + block_size - 1) / block_size;
    dim3 thr(B, B);

    for (int kb = 0; kb < num_blocks; ++kb) {
        int k_start = kb * block_size;

        // Phase 1
        phase1<B, R><<<1, thr>>>(n, k_start, graph);

        if (num_blocks > 1) {
            phase2_row<B, R><<<num_blocks, thr>>>(n, kb, graph);
            phase2_col<B, R><<<num_blocks, thr>>>(n, kb, graph);
            phase3<B, R><<<dim3(num_blocks, num_blocks), thr>>>(n, kb, graph);
        }
    }
}

