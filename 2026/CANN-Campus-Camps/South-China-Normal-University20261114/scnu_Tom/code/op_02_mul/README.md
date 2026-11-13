# op_02_mul

提交团队: 华南师范大学
提交者: ljt


## 算子实现介绍

本算子基于Ascend C编程语言原生开发，实现矢量乘法功能（z = x * y）。具体实现策略如下：
单核调度实现：根据赛题约束，在Host侧Tiling中配置 SetBlockDim(1)，采用单核模式启动Kernel，由单一AI Core完成全量数据的搬运与计算，降低了多核切分的复杂度，保证了算子执行的稳定性。
UB切分与32B对齐：在Kernel侧Tiling中，严格按照硬件要求进行32字节（BLOCK_SIZE）对齐处理。根据输入数据类型动态计算对齐元素个数（alignElements = 32B / sizeof(T)），确保Global Memory到Local Tensor的搬运长度及UB内存分配均严格为32B的整数倍，避免非对齐访问。

Tiling策略：Tiling策略将总数据量 totalLength 向上对齐到 BLOCK_SIZE 的整数倍。在Kernel侧，将对齐后的总长度切分为多个块（tileNum）进行循环处理，配合双缓冲（Double Buffering）机制（BUFFER_NUM=2），流水化执行 CopyIn、Compute 与 CopyOut，实现计算与搬运的并行，提升算子性能。
核心计算与多类型支持：采用C++模板类（template &lt;class DT_X&gt;）设计，完美适配Host侧TilingKey的模板申明。在 Compute 阶段调用 AscendC::Mul 接口实现逐元素乘法，原生支持 float32 与 float16 两种数据类型。
