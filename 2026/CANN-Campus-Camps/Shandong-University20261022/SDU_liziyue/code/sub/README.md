# sub

提交团队: 山东大学
提交者: 黎子月


## 算子实现介绍

本算子基于 Ascend C 编程语言实现矢量逐元素减法，计算公式为：
\[
z_i = x_i - y_i
\]
输入张量 x、y 和输出张量 z 的形状均为 (8, 2048)，数据格式为 ND，支持 float16 和 float32 两种数据类型。
算子采用多核并行与分块计算方式。Host 侧首先根据输入张量形状计算总元素数，并将任务划分到 8 个 Vector Core 上。每个核负责处理 2048 个元素，并进一步划分为多个数据块，以减少单次计算所需的片上内存空间。
Kernel 侧按照 CopyIn → Compute → CopyOut 的流程执行：
1. CopyIn：使用 DataCopy 将输入数据从 Global Memory 搬运到 Unified Buffer。
2. Compute：调用 AscendC::Sub 完成两个输入张量的逐元素减法。
3. CopyOut：将计算结果从 Unified Buffer 搬运回 Global Memory。
实现中使用 TPipe 和 TQue 管理输入、输出队列，并配置双缓冲机制，使数据搬运与矢量计算能够以流水方式执行，提高硬件资源利用率。通过 C++ 模板分别实例化 half 和 float 类型的 Kernel，并根据输入数据类型在 Host 侧选择对应的核函数。
该实现覆盖全部 8 × 2048 = 16384 个元素，计算过程中不改变张量形状和数据类型，最终输出与标准表达式 x - y 保持一致。
