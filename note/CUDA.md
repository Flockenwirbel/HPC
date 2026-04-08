# CUDA

## GPU 硬件组织

GPU 由多个 SM (Streaming Multiprocessor) 组成, 每个 SM 包含多个 CUDA Core; SM 内的线程被分组成 Warp (32 个线程), 同一 Warp 内的线程执行相同的指令但操作不同的数据 (SIMT), 这些线程分布在 SM 的多个 Core 上并行执行, 是 GPU 实现并行计算的基本机制.

GPU 与 CPU 的区别在于 CPU 是少量大核, 单核计算能力强; GPU 是大量小核, 适合大规模并行计算.

## 在 GPU 上设计高效编程模型

CUDA: Compute Unified Device Architecture

```cpp
__global__ void my_kernel() {
    // ...
}

int main() {
    // ...
}
```

线程被划分到线程块中, 这是与 SM 的架构相匹配的. Thread block 的执行顺序不能有任何假设, 底层硬件会对线程块以任意顺序进行调度执行. 线程块之间没有依赖, 具有好的扩展性, 核函数可以扩展到任意的 SM 上.

Compute Capability 是 NVIDIA 定义的 GPU 架构版本, 不同版本支持不同的功能和性能特性. 例如, Compute Capability 3.0 引入了动态并行, 允许 GPU 内部的线程启动新的线程块; Compute Capability 5.0 引入了更大的共享内存和更高的并行度. 在 Compute Capability 3.0 之后, GPU 支持更高的并行度和更大的共享内存.

GPU 的内存层次结构包括全局内存 (Global Memory), 共享内存 (Shared Memory) 和寄存器 (Registers). 全局内存访问延迟较高, 共享内存和寄存器访问速度较快. 

Thread ID / Block ID. `threadIdx.x`, `threadIdx.y`, `threadIdx.z` 表示线程在块内的索引; `blockIdx.x`, `blockIdx.y`, `blockIdx.z` 表示块在网格中的索引; `blockDim.x`, `blockDim.y`, `blockDim.z` 表示每个块中线程的数量. 通过这些索引, 每个线程可以计算出自己在整个网格中的唯一 ID, 从而访问不同的数据元素进行并行计算.

eg. 

```cpp
dim3 grid (3, 2); // 3 blocks in x, 2 blocks in y
dim3 blk(5, 3); // 5 threads in x, 3 threads in y
my_kernel<<<grid, blk>>>(); // Launch kernel with 3x2 blocks and 5x3 threads per block. Each thread has a unique ID based on its block and thread indices.
```

一个程序示例.

```cpp
// 1-dim.
__global__ void VecAdd(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    VecAdd<<<1, N>>>(A, B, C); // Launch kernel with 1 block and N threads. Each thread computes one element of the output vector C by adding corresponding elements from A and B.
}
```

```cpp
// 2-dim.
__global__ void MatAdd(float *A, float *B, float *C) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main() {
    dim3 threadsPerBlock(N, N);
    MatAdd<<<1, threadsPerBlock>>>(A, B, C); // Launch kernel with 1 block and N x N threads. Each thread computes one element of the output matrix C by adding corresponding elements from A and B.
}
```

块内的线程数有 1024 个的限制. 可以设置更多线程块来提高总线程数.

```cpp
// 2-dim with multiple blocks.
__global__ void MatAdd(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Calculate global column index
    C[i][j] = A[i][j] + B[i][j]; // Each thread computes one element of the output matrix C by adding corresponding elements from A and B.
}

int main() {
    dim3 threadsPerBlock(16, 16); // 16 x 16 threads per block
    dim3 numBlocks(N / 16, N / 16); // Calculate number of blocks needed to cover the entire N x N matrix
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C); // Launch kernel with calculated number of blocks and threads per block.
}
```

线程块内所有线程不保证快慢顺序, 但可以通过 `__syncthreads()` 进行同步, 确保所有线程都执行到该点后再继续执行后续代码. 线程块之间的线程不能同步.

形如 `__device__`, `__host__`, `__global__` 等的变量类型修饰符可以自行查阅.

含有 CUDA 语言的源文件都需要用 NVCC 编译器进行编译.

GPU 端两阶段编译过程:

- 设备无关编译: `nvcc --ptx -o a.ptx a.cu`
- 设备相关编译: `nvcc -gencode arch=compute_30,code=sm_30 -o a.o a.ptx`

## 简单的 CUDA 程序

```cpp
__global__ void mykernel(void) {
    printf("Hello, CUDA!\n");
}

int main(void) {
    mykernel<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to finish before exiting
    return 0;
}
```

```cpp
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}
int main(void) {
    int a, b, c; // host copies of a, b, c.

    // add<<<1, 1>>>(&a, &b, &c); // device copy of add, and execution configuration. But this will likely fail because a, b, c are not allocated on the device.

    int *d_a, *d_b, *d_c; // device copies of a, b, c

    cudaMalloc((void **)&d_a, sizeof(int)); // allocate space for a on the device
    cudaMalloc((void **)&d_b, sizeof(int)); // allocate space for b on the device
    cudaMalloc((void **)&d_c, sizeof(int)); // allocate space for c on the device

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice); // copy a to device
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice); // copy b to device
    add<<<1, 1>>>(d_a, d_b, d_c); // launch add() kernel on GPU with 1 block and 1 thread

    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost); // copy result back to host

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); // free device memory

    printf("Result: %d\n", c); // print the result
    return 0;
}
```

注意 CPU 上和 GPU 上的内存是分开的, 需要使用 `cudaMalloc` 和 `cudaMemcpy` 来在 GPU 上分配内存和传输数据.

GPU 最好处理连续的内存, 因此应当 `index = blockIdx.x * blockDim.x + threadIdx.x` 来计算全局线程 ID, 以便每个线程处理连续的数据元素.