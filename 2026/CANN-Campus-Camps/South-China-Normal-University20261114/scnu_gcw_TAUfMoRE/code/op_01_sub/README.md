# op_01_sub

提交团队: 华南师范大学
提交者: gcw_TAUfMoRE


## 算子实现介绍

基于 Ascend C 的昇腾 NPU 矢量减法算子实验介绍
一、 实验背景与目的
本次实验基于华为昇腾 CANN 架构，采用 Ascend C 编程语言进行算子原生开发。要求在昇腾 NPU 硬件上实现一款基础的双目矢量减法算子（z = x - y），支持 (8, 2048) 维度的张量输入，并兼容 float32 与 float16 两种数据类型。旨在通过实战掌握 Ascend C 的编程范式及 NPU 硬件加速原理。

二、 核心实现方案
实验采用经典的 Host-Tiling-Kernel 架构，通过 Host 侧解析 Tensor 信息（Shape、Dtype）并下发 Tiling 参数，驱动 Device 侧核函数执行。

多核并行切分：将 16384 个总元素均分至多个 AI Core，最大化利用 NPU 的并行计算能力。

核内流水线设计：利用 TPipe 和 TQue 队列机制，开启 BUFFER_NUM=2 的双缓冲（Double Buffer）。实现 CopyIn（DMA搬运）、Compute（Vector计算）和 CopyOut（DMA搬出）三级流水线并行，有效掩盖数据搬运带来的内存延迟。

类型泛化与模板化：利用 C++ 模板技术（template &lt;typename T&gt;），通过 dtype 标志位在 Host 和 Kernel 侧动态实例化 float 与 half 分支，实现一套代码兼容不同精度的需求。

三、 关键技术难点攻克

32字节对齐约束：严格遵守 UB（Unified Buffer）内存的 32 字节对齐规则，计算单次切块的 Tile 长度，确保起始地址和搬运长度合规。

尾块内存安全：针对无法被核数或对齐长度整除的边界情况，使用 DataCopyPad 进行带填充的安全搬运，彻底杜绝内存读写越界。

四、 实验意义
通过本次实验，不仅掌握了 Ascend C 核心 API（如 DataCopy、Sub、TQue）的使用，更深入理解了昇腾 NPU 的“计算与搬运分离”架构特性，为后续开发更复杂的高性能自定义算子打下了坚实基础。
