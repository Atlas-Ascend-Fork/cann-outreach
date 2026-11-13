# op_03_tanh

提交团队: 华南师范大学
提交者: zhangdu


## 算子实现介绍

一、 算子背景与功能
本算子基于 Ascend C 编程语言进行原生开发，在昇腾 NPU 硬件上实现了 Tanh（双曲正切）激活函数。
数学公式：y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
输入/输出规格：输入张量 x 与输出张量 y 的形状均为 (8, 2048)，支持 float32 与 float16 数据类型，数据排布格式为 ND。
二、 Host 侧实现（Tiling 与推导）
Host 侧负责算子的图推导与 Tiling 切分策略，主要由以下三个模块构成：
算子原型注册（tanh_def.cpp）：定义单输入 x、单输出 y 的算子原型，配置支持的数据类型与 ND 格式。
形状与类型推导（tanh_infershape.cpp）：由于 Tanh 为逐元素（Element-wise）算子，实现中直接获取输入 x 的 Shape 与 DataType，并推导赋值给输出 y，保证两者完全一致。
Tiling 策略（tanh_tiling.cpp）：
遍历输入张量的维度，计算得出总元素个数（totalLength = 16384）。
结合昇腾 NPU 硬件架构，配置启动 8 个 AI Core（SetBlockDim(8)）以实现多核并行计算。
将总长度 totalLength 与切分参数 tileNum（设为 8）打包为 TanhTilingData 结构体，下发给 Device 侧的 Kernel 函数。
三、 Kernel 侧实现（NPU 核心计算）
Kernel 侧代码位于 op_kernel/tanh.h 与 tanh.cpp，采用标准的三段式流水线架构，以最大化 NPU 的吞吐性能：
多核切分：通过 AscendC::GetBlockIdx() 获取当前核的 ID，结合 Host 侧下发的数据，计算当前核在 Global Memory（GM）上的起始偏移量与处理长度，实现 8 核无冲突并行计算。
双缓冲流水线（CopyIn -&gt; Compute -&gt; CopyOut）：
设置 BUFFER_NUM = 2，开启双缓冲（Double Buffer）机制
CopyIn：利用 DataCopy 将数据分块搬运到 Local Memory（UB）。
Compute：从队列取出数据，直接调用 Ascend C 硬件原语 AscendC::Tanh 完成激活计算。
CopyOut：将计算结果从 UB 搬回 GM。
性能与精度优化（核心亮点）：
精度稳定：没有采用常规的泰勒展开或 exp 公式拼接去模拟计算，而是直接调用底层单指令 AscendC::Tanh，彻底避免了 float16 在极端数值下使用 exp 带来的精度溢出和 NaN 问题。
极致性能：省去了中间临时变量（如 exp(x)、分子分母）的 Buffer 申请与释放过程，配合双缓冲流水线，使得 MTE2（数据搬运）与 Vector（向量计算）单元完美重叠，实现“又稳又快”的算子执行。
