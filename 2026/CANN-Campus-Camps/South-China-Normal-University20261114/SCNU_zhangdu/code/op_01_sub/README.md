# op_01_sub

提交团队: 华南师范大学
提交者: zhangdu


## 算子实现介绍

本算子基于 Ascend C 编程语言原生开发，在昇腾 NPU 硬件上实现矢量减法（Vector Sub）功能。

一、 功能与规格

计算公式：z = x - y （逐元素相减）

输入规格：张量 x、y，维度形状均为 (8, 2048)，数据类型支持 float32 / float16，格式为 ND。

输出规格：张量 z，维度形状为 (8, 2048)，数据类型与输入保持一致，格式为 ND。

二、 Host 侧实现（Tiling 与推导）
Host 侧代码主要完成 Tiling 策略和推导逻辑：

InferShape（形状推导）：因为逐元素减法不改变张量维度，直接获取输入 x 的 Shape 并赋值给输出 z。

InferDataType（类型推导）：获取输入 x 的数据类型（float32 或 float16），将其设置为输出 z 的数据类型，确保精度一致。

Tiling 切分策略：

通过 GetInputShape 遍历获取输入张量的总元素个数（totalLength = 8 * 2048 = 16384）。

结合硬件特性，配置启动 8 个 AI Core（context-&gt;SetBlockDim(8)）进行并行计算，每个核负责处理 2048 个元素。

将总长度（totalLength）和分块数（tileNum）写入 Tiling 结构体传递给 Kernel 侧。

三、 Kernel 侧实现（NPU 流水线）
Kernel 侧核函数核心逻辑继承并改造自模板：

多核并行切分：通过 AscendC::GetBlockIdx() 获取当前核 ID，结合 Host 侧下发的数据，计算当前核在 Global Memory（GM）上的数据起始偏移量和处理长度。

三段式流水线（CopyIn -&gt; Compute -&gt; CopyOut）：

CopyIn：使用 DataCopy 指令将 GM 上的数据分块搬入 Local Memory（UB）的 VECIN 队列中。

Compute：从队列中取出数据，调用 Ascend C 向量指令 AscendC::Sub 完成逐元素减法计算，并将结果写入 VECOUT 队列。

CopyOut：使用 DataCopy 指令将计算结果从 UB 搬运回 GM 的输出张量中。

双缓冲优化：设置 BUFFER_NUM = 2 开启双缓冲（Double Buffer）机制。使得数据搬运和向量计算指令能够重叠执行，有效掩盖数据搬运延迟，提升算子吞吐量。

四、 核心修改点（基于参考代码）

将算子注册的类名、核函数入口名由 Add 修改为 Sub。

将 AddTilingData 结构体更名为 SubTilingData，保持参数（totalLength, tileNum）不变。

将 Kernel 侧核心计算指令由 AscendC::Add 替换为 AscendC::Sub，实现减法逻辑。
