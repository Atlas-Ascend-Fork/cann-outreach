# op_02_mul

提交团队: 华南师范大学
提交者: HAJIMIO134


## 算子实现介绍

Tiling 切分策略

多核并行：Host 侧根据环境配置分配计算核心数（SetBlockDim(8)），按核数将总数据量 totalLength 均分为各核的处理长度 blockLength。

单核切分：单核内将数据进一步切分为 tileNum=8 个数据块，并结合双缓冲 BUFFER_NUM=2 计算最终的单次搬运粒度 tileLength。

UB 内存与流水线设计

采用 AscendC::TPipe 统一管理 Unified Buffer (UB) 内存。

使用 TQue 分别管理输入输出队列，通过 InitBuffer 分配双缓冲空间。

严格按照 CopyIn（搬入） -&gt; Compute（计算） -&gt; CopyOut（搬出） 的三级流水线执行，通过 EnQue / DeQue 控制多指令同步，实现数据搬运与矢量计算的高效并行。

计算核心

通过模板化设计 KernelMul&lt;DT_X&gt; 适配 float 与 half 等多种数据类型。

在 Compute 阶段，核心调用 AscendC::Mul 硬件指令，完成单块 LocalTensor 的逐元素矢量相乘，最终搬出至 Global Memory。
