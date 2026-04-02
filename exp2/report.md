# Ring Allreduce 算法实验报告

刘家豪 2024010779

## 算法简述

Ring Allreduce 算法的执行过程分为两部分: $P - 1$ 步 Reduce-Scatter 和 $P - 1$ 步 Allgather. 

在开始之前, 首先将 `recvbuf` 的内容初始化为 `sendbuf` 的内容, 因为 Ring Allreduce 是在 `recvbuf` 上原地累加的. 然后将数据分为 $P$ 块, 每块大小为 $\dfrac{n}{comm\_sz}$.

在 Reduce-Scatter 阶段的第 $i$ 步, 进程 `my_rank` 将自己的第 `my_rank - i + comm_sz) % comm_sz` 块数据发送给进程 `(my_rank + 1) % comm_sz`, 并接收来自 `(my_rank - 1 + comm_sz) % comm_sz` 进程的 第 `(my_rank - i - 1 + comm_sz) % comm_sz` 块数据进行累加. 使用 `MPI_Sendrecv` 进行收发. `recvbuf` 不能和要累加的目标地址相同, 需要先收到临时 `buffer`, 再自己做累加循环. 

在 Allgather 阶段的第 $i$ 步, 进程 `my_rank` 将自己的第 `(my_rank + 1 - i + comm_sz) % comm_sz` 块数据发送给进程 `(my_rank + 1) % comm_sz`, 并接收来自 `(my_rank - 1 + comm_sz) % comm_sz` 进程的第 `(my_rank - i + comm_sz) % comm_sz` 块数据. 使用 `MPI_Sendrecv` 进行收发. 由于数据将被覆盖, `recvbuf` 可以直接作为发送和接收的缓冲区.

## 通信时间

MPI_Allreduce:   2429.61 ms.
Naive_Allreduce: 4767.3 ms.
Ring_Allreduce:  2143.03 ms.