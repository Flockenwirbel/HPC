# 测量 OpenMP 并行 for 循环不同调度策略的性能 实验报告

刘家豪 2024010779

## 线程数

根据 `srun -N 1 -c 28 ./omp_sched`, 线程数为 28.

## 补全指导语句

在 `omp_sched.c` 的 line 59 和 line 68 处分别补全

```cpp
#pragma omp parallel for schedule(<strategy>)
```

其中 `<strategy>` 分别替换为 `static`, `dynamic`, `guided`.

## 结果分析

| | Static | Dynamic | Guided |
|---|---|---|---|
|Uniform| 68.4492 ms | 77.1465 ms | 69.2513 ms |
|Random| 190.923 ms | 167.979 ms | 165.621 ms |

对于工作量均匀的序列, Static 凭借预分配机制取得了 68.45 ms 的最佳表现, Dynamic 则由于反复的同步开销导致了性能下降. 而针对工作量波动剧烈的随机序列, Static 固定的分配方式导致了严重的负载不均衡, 此时 Dynamic 和 Guided 策略通过动态调整块大小, 既有效降低了获取任务的同步频率, 又实现了优异的负载平衡, 最终在随机场景下 Guided 以 165.62 ms 的成绩优于 Dynamic 和 Static.