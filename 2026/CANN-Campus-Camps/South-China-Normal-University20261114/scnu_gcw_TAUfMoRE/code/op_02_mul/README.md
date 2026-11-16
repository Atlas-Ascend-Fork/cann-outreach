# op_02_mul

提交团队: 华南师范大学
提交者: Clover


## 算子实现介绍

本次实验基于 Ascend C 编程语言，在昇腾 NPU（Ascend 910B）硬件上实现一款逐元素向量乘法算子（z = x * y）。
核心任务：
• 输入输出：处理两个输入张量 x、y 和一个输出张量 z，形状均为 (8, 2048)，支持 float16 和 float32 数据类型。
• Host侧开发：完成 Tiling 切分策略（多核任务分配）、Workspace 配置，以及输出张量的 Shape 和 DataType 推导。
• Kernel侧开发：完成核函数初始化（Init），并实现标准的 Ascend C 流水线编程：数据搬入（CopyIn） -&gt; 向量乘法计算（Compute，调用 Mul 接口） -&gt; 结果搬出（CopyOut）。
实验目的：
通过补齐模板代码，掌握昇腾算子开发的基础流程，理解 NPU 上的多核切分机制与片上内存（UB）的流水线调度原理。
