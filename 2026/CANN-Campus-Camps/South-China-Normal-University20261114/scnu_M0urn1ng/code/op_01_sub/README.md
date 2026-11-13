# op_01_sub

提交团队: 华南师范大学
提交者: 曾理


## 算子实现介绍

本算子基于 Ascend C Kernel 直调实现矢量减法 z = x - y，支持 float32 / float16，ND 格式，输入输出张量形状均为 (8, 2048)。

实现策略：采用 8 核多核并行。host 侧 run_kernel 从 TensorGroupInfo 推导 totalLength，并按输入 dtype 实例化 float / half 核函数；各 AI Core 通过 SetGlobalBuffer 以 GetBlockIdx() * blockLength 偏移切分 Global Memory，使用 TPipe / TQue（VECIN、VECOUT）与 DataCopy 完成 GM 与 UB 之间的数据搬运，按 CopyIn → Compute → CopyOut 三段流水执行，计算调用 AscendC::Sub，AllocTensor 与 FreeTensor 严格配对，避免 UB 泄漏。

Tiling 策略：Tiling 结构体包含 totalLength 与 tileNum 两个字段；blockLength = totalLength / GetBlockNum()，tileLength = blockLength / tileNum / BUFFER_NUM。本例 totalLength = 16384、blockDim = 8、tileNum = 8、BUFFER_NUM = 2，得到 blockLength = 2048、tileLength = 128，单核循环 16 次；每块数据 512 B（fp32）/ 256 B（fp16），均为 32 B 整数倍，满足 UB 对齐要求。队列深度 QUEUE_DEPTH = 1，BUFFER_NUM = 2 用于 Double Buffer 内存分配。

验证结果：CANNJudge 验证集 2/2 测试点 PASS，输出错误占比 0.00%，用时 6.10 μs / 6.00 μs。
