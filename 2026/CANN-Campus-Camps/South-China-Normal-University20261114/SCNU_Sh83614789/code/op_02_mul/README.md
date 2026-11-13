# op_02_mul

提交团队: 华南师范大学
提交者: Sh83614789


## 算子实现介绍

采用 Ascend C 三级流水（CopyIn / Compute / CopyOut）实现 z = x * y，核函数模板化，通过 tiling key 按输入 dtype（float32 / float16）分派实例。

Tiling 策略：host 侧遍历输入 x 的 StorageShape 各维相乘得到 totalLength，tileNum 固定为 8，并通过 SetBlockDim(8) 配置 8 个 AICore 并行。

多核切分：每个核处理 blockLength = totalLength / GetBlockNum() = totalLength / 8 个元素，通过 SetGlobalBuffer 按 GetBlockIdx()*blockLength 偏移划分 x、y、z 的 GM 区间，保证各核数据不重叠、全覆盖。

片内切块：单核内再按 tileNum = 8 分块，配合 BUFFER_NUM = 2 的双缓冲将每块长度折半为 tileLength = blockLength / 8 / 2。以输入形状 8×2048 为例，每核 2048 个元素、每块 128 个元素，float 对应 512B、half 对应 256B，均为 32B 的整数倍，满足 UB 搬运对齐要求，无尾部残余数据。

计算：每块用 AscendC::Mul 对 128 个元素的 LocalTensor 做矢量乘，单核 16 次搬入-计算-搬出循环覆盖全部数据；输入 Tensor 计算完成后立即 FreeTensor，避免 UB 内存泄漏。
