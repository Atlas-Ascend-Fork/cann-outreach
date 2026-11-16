# op_02_mul

提交团队: 华南师范大学
提交者: gcw_0S56kdyM


## 算子实现介绍

Mul 算子是一个逐元素二元乘法算子，接收两个数据类型与 shape 相同的输入张量 x 和 y，对应元素相乘后输出同 shape、同数据类型的张量 z（即 z[i] = x[i] * y[i]），支持 float32 与 float16 两种数据类型、ND 格式，属于典型的向量逐元素计算（ElementWise）算子；其 Host 侧 Tiling 根据输入元素总数与平台 AIV 核数计算分核方案并填充 tiling 数据，Kernel 侧采用多核并行、双缓冲（BUFFER_NUM=2）的三段式流水（CopyIn-Compute-CopyOut）将数据从 Global Memory 搬入 Unified Buffer、调用 AscendC::Mul 完成逐元素乘法后再搬回 Global Memory，从而在昇腾 AI 处理器上高效完成大规模张量的逐元素相乘。
