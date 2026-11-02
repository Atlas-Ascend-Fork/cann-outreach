# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: gcw_7Xs2liFP


## 算子实现介绍

本算子基于Ascend C编程语言开发，实现矢量减法（z = x - y）。支持 float32 与 float16 数据类型，数据格式为 ND。

实现策略：

多核并行：采用 8 个 AI Core (blockDim=8) 进行并行计算，按数据首地址偏移均分总数据量 (8*2048)，每个核处理 2048 个元素。

Tiling 与 Double Buffer 切分：每个核内设置 tileNum=8 与 BUFFER_NUM=2。通过 tileLength = blockLength / tileNum / BUFFER_NUM 计算得出每次搬运 128 个元素。利用 Double Buffer 机制实现数据搬入、计算、搬出的流水线并行，隐藏搬运时延。

内存对齐与队列管理：使用 TPipe 和 TQue 进行内存管理，利用 VECIN 和 VECOUT 队列控制数据流。在分配 LocalMemory 队列时，严格按照 tileLength * sizeof(T) 进行 32B 对齐的 UB 切分。

核心计算：在 Compute 阶段，通过 AscendC::Sub 接口完成向量减法操作，配合 FreeTensor 及时释放内存防止队列阻塞。
