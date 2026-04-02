# OpenMP Model

- 共享内存: 所有处理单元与共享内存相连, 处于同一个地址空间下, 任何处理单元可通过地址直接访问任意内存位置
- 内存访问架构: 均匀 (UMA) 或非均匀 (NUMA)

## 概述

OpenMP: Open Multi-Processing, 显式指导多线程, 共享内存并行 API
- 提供轻量级, 可移植的语法标准
- 增量式并行化: 在串行代码基础上, 添加几行, 转换为并行代码
  `#pragma omp parallel`
- 需要编译器支持: C/C++/Fortran

- OpenMP 不会自动并行化代码, 需要程序员使用编译指令 (pragmas) 明确指定并行区域和工作分配
- OpenMP 不保证加速
- OpenMP 无法自动避免数据竞争和死锁等并发问题, 需要程序员正确使用同步机制

## 组成

- ~80%: 编译器预处理指导语句
- ~19%: 函数调用
- ~1%: 环境变量

```cpp
int main() {
    omp_set_num_threads(16); // 设置线程数
    #pragma omp parallel
    {
        // 这里的代码将被多个线程执行
        printf("Hello from thread %d\n");
    }
    return 0;
}
```

### 编译器预处理指导语句

- `#pragma omp`: 必须
- `parallel, for` 等: 指导语句名
- 子句, 可选, 若干个, 无顺序
- 换行: 必须

- Fork-Join 模型: 主线程创建多个工作线程, 执行并行代码, 结束后销毁工作线程, 返回主线程继续执行串行代码
- 类比 `pthread`: `#pragma omp parallel` 类似于 `pthread_create`, 但 OpenMP 负责线程管理和调度, 不需要显式创建和销毁线程

### 并行指导语句

- 并行区结构体: 创建一组 OpenMP 线程, 执行并行代码块
  - `#pragma omp parallel`
  - 子句包括:
    - `private`
    - `firstprivate`
    - `shared`
    - `copyin`
    - ...
  - 所有线程均执行并行区内的代码
  - 并行区结束时所有线程会同步, 使用 `nowait` 可避免同步
  - 并行区需要为函数内的一个语句块, **不能 `goto` 跳转出入并行区**, 但是并行区内可以调用其他函数
  - 线程数量可以使用多个方式修改, 优先级: `#pragma` > 函数调用 > 环境变量
    - `num_threads(10)`
    - `omp_set_num_threads(10)`
    - `OMP_NUM_THREADS=10` 环境变量
    - `if (para == true)`
  - 除非启用嵌套并行区, 内层并行区只用一个线程
    - `omp_set_nested(1)` 启用嵌套并行区
    - `omp_get_nested()` 获取嵌套并行区状态
- 任务分配结构体: 将代码区中的任务自动分配给不同线程执行
  - 不创建新线程, 进入任务分配结构体时各个线程不同步, 离开时有隐式同步
  - 典型任务分配结构体
    - `DO / for`
    - `sections`
    - `single`
  - `#pragma omp for`
    - 子句包括
      - `private`
      - `firstprivate`
      - `lastprivate`
      - `linear`
      - `schedule`
      - ...
    - 把 `parallel` 和 `for` 分开写和写在一起是等价的
    - `schedule` 子句, 只会对外层展开, 如果要展开内层, 需要添加 `collapse` 子句
      - `static`: 循环迭代平均分配给线程, 适合负载均衡的循环
      - `dynamic`: 循环迭代动态分配给线程, 线程执行完之后再请求一个, 迭代与线程的对应关系不对应, 引入额外开销
      - `guided`: 与 `dynamic` 类似, 但每次分配的迭代数量逐渐减少
      - `runtime`: 由环境变量 `OMP_SCHEDULE` 决定
      - `auto`: 由编译器和运行时系统决定
  - `#pragma omp sections`
    - 子句包括
      - `private`
      - `firstprivate`
      - `lastprivate`
      - `linear`
      - ...
    - 将代码区分成若干个独立的部分, 每个部分由一个线程执行, 适合不同任务的并行化
  - `#pragma omp single`
    - 子句包括
      - `private`
      - `firstprivate`
      - `lastprivate`
      - `linear`
      - `nowait`
      - ...
    - 只有一个线程执行代码区, 其他线程跳过, 适合需要串行执行的代码, 但又不想阻塞其他线程的情况

## 数据共享与线程同步

对于共享内存编程模型, 大多数变量默认共享, 作用域内变量可以显式控制

- `private`: 私有变量, 进入前不会初始化, 离开后销毁
- `firstprivate`: 私有变量, 进入前会初始化为原变量的值, 离开后销毁
- `lastprivate`: 私有变量, 进入前不会初始化, 离开后将最后一个迭代的值赋给原变量

对存在数据竞争的变量访问需要保护. 以下代码是不正确的:

```cpp
float dot_prod(float* a, float* b, int N) {
    float sum = 0.0;
    #pragma omp parallel num_thread(2)
    #pragma omp for shared(sum)
    for (int i = 0; i < N; ++i) {
        sum = sum + a[i] * b[i];
    }
    return sum;
}
```

需要在 sum 行之前添加 `#pragma omp critical` 来保护对 sum 的访问, 但这会引入大量的同步开销. 

另外有 `barrier`, 等待所有线程到达, 适合需要全局同步的情况; `atomic`, 适合对单个变量的原子操作.

锁函数 `omp_set_lock`, `omp_unset_lock`, 适合需要更复杂的同步机制的情况. 

当需要获取一个变量的最新值时, 需保证变量被 `flush`, `#pragma omp flush <memory_order> <var_list>`. 自动进 `flush` 的命令包括 `parallel` (I/O), `critical` (I/O), `ordered` (I/O), `for` (O), `sections` (O), `single` (O)

标准的做法是 `reduce` 子句, 适合需要对变量进行归约操作的情况. 例如:

```cpp
float dot_prod(float* a, float* b, int N) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum = sum + a[i] * b[i];
    }
    return sum;
}
```

使用 `reduction` 性能最优, `critical` 和 `atomic` 性能非常差.

## 编译与运行

`gcc -g -Wall -fopenmp -o main main.c`

`-f` 仅在 gcc, g++, g77, gfortran 中使用, Intel icc 是 `-qopenmp`, PGI pgcc, pgCC, pgf77, pgf90 是 `-mp`, Clang clang, clang++ 是 `-fopenmp`.

`OMP_NUM_THREADS=16 ./main`

## 常用的库函数与环境变量

自己去查吧!

## MPI-OpenMP

MPI 是 "汇编", 需要手动控制所有的通信和同步, 而 OpenMP 是 "高级语言", 只需要添加几行代码就可以实现并行化.

二者可以混合编程, 机器之间 MPI, 机器内部 OpenMP.