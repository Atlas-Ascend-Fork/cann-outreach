# op_01_sub

提交团队: 华南师范大学
提交者: Sh83614789


## 算子实现介绍

采用 Ascend C 三级流水（CopyIn / Compute / CopyOut）实现 z = x - y，核函数模板化，同时支持 float32 与 float16，host 侧按输入的 dtype 分派对应实例。

多核切分：blockDim 固定为 8，每核处理 blockLength = totalLength / GetBlockNum() = 2048 个元素（输入形状 8×2048，恰好每核一行），通过 SetGlobalBuffer 按 GetBlockIdx()*blockLength 偏移划分 x、y、z 的 GM 区间。

Tiling 策略：单核内再按 tileNum = 8 切块，并配合双缓冲 BUFFER_NUM = 2 将每块长度折半为 tileLength = 128 个元素。float 对应 512B、half 对应 256B，均是 32B 的整数倍，满足 UB 搬运对齐要求，无尾部残余数据。

计算：每块用 AscendC::Sub 对 128 个元素的 LocalTensor 一次性做矢量减法，单核 16 次搬入-计算-搬出循环覆盖全部 2048 个元素，队列用 QUEUE_DEPTH = 1 保证内存不溢出。
