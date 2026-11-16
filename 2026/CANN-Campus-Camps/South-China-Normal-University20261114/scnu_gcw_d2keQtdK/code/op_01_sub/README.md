# op_01_sub

提交团队: 华南师范大学
提交者: gcw_d2keQtdK


## 算子实现介绍

这是一个基于 Ascend C 编程语言，在昇腾 NPU 上实现的矢量减法算子
简短来说，它的核心实现逻辑分为三步：
多核并行切分：根据 NPU 的核心数（blockDim），将总长度的数据平均分配给各个核，并通过 GetBlockIdx() 计算每个核在全局内存（Global Memory）中的起始偏移量。
流水线分块处理（Double Buffer）：在每个核内部，数据被进一步切分成小块（Tile），利用 TPipe 和队列（Queue）实现 CopyIn -&gt; Compute -&gt; CopyOut 的流水线作业，隐藏数据搬运的延迟。
矢量计算：在 Compute 阶段，将数据从全局内存搬到片上内存（UB）后，调用 Ascend C 提供的内置矢量指令 AscendC::Sub 完成减法计算，最后再搬回全局内存。
