# exp0 实验报告

刘家豪 2024010779

## `pow_a` 的源代码

### 在 `openmp_pow.cpp` 中

```cpp
void pow_a(int *a, int *b, int n, int m) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        b[i] = x;
    }
}
```

### 在 `mpi_pow.cpp` 中

```cpp
void pow_a(int *a, int *b, int n, int m, int comm_sz /* 总进程数 */) {
    // TODO: 对这个进程拥有的数据计算 b[i] = a[i]^m
    for (int i = 0; i < n / comm_sz; ++i) {
        b[i] = 1;
        for (int j = 0; j < m; ++j) {
            b[i] *= a[i];
        }
    }
}
```

## `openmp` 版本的加速情况

|线程数目|运行时间 ($\mu\text{s}$)|加速比|
|---|---|---|
|$1$|$14014441$|/|
|$7$|$2009362$|$6.97$|
|$14$|$1014949$|$13.81$|
|$28$|$509916$|$27.50$|

## `mpi` 版本的加速情况

|进程数目|运行时间 ($\mu\text{s}$)|加速比|
|---|---|---|
|$1\times 1$|$14007170$|/|
|$1\times 7$|$2005660$|$6.98$|
|$1\times 14$|$1003576$|$13.95$|
|$1\times 28$|$504556$|$27.77$|
|$2\times 28$|$411939$|$34.01$|