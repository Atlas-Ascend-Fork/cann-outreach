# op_01_sub

提交团队: 华南师范大学
提交者: HAJIMIO134


## 算子实现介绍

本算子基于 Ascend C 编程语言原生开发，在昇腾 NPU 上实现矢量减法（z = x - y）。具体实现策略如下：

核函数实现与计算逻辑

采用模板化设计（template &lt;typename T&gt;），支持 half (fp16) 与 float (fp32) 等多种数据类型。

核心计算调用 Ascend C 矢量指令 AscendC::Sub，实现单个 Tile 内的逐元素相减。

Tiling 切分策略

按 BLOCK_SIZE 对齐：Host 侧根据 totalLength 计算分块，确保切分粒度严格对齐到 BLOCK_SIZE（即 32B 对应的元素个数，如 fp16 为 16 个元素，fp32 为 8 个元素）的整数倍，以此满足底层硬件的搬运效率。

针对不满足 32B 整除的尾部数据，利用 DataCopyPad 进行补齐搬运，确保边界数据不丢失。

UB 内存切分与流水线设计

内存管理：使用 AscendC::TPipe 统一管理 UB 内存，通过 AscendC::TQue 分别管理输入与输出队列，队列深度设为 1（或根据硬件配置设为 2 开启双缓冲）。

UB 32B 对齐：InitBuffer 分配 LocalTensor 内存时，依据 32B 对齐原则计算所需字节数。

流水线执行：采用 CopyIn -&gt; Compute -&gt; CopyOut 的标准三级流水线模式。单核内通过 for 循环逐个 Tile 搬运和计算，通过 EnQue 与 DeQue 控制同步，保障计算与数据搬运的高效协同。

单核/多核适配

支持单核直接处理全量数据；如申请多核（blockDim &gt; 1），则按 blockLength 将 totalLength 均分至各核，各核通过 GetBlockIdx() 计算偏移地址，独立进行 Tile 循环计算。
