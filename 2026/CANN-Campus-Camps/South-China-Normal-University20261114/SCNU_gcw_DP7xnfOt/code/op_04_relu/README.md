# op_04_relu

提交团队: 华南师范大学
提交者: 何建基


## 算子实现介绍

使用 Ascend C 实现 ReLU 激活算子 y=max(0,x)。Host 侧按输入 shape 计算总元素数，根据 dtype 计算 32B 对齐的 blockFactor，并按实际使用核数设置 blockDim；同时结合 UB 容量计算 ubFactor。Kernel 侧 multi-core 切分后按 UB 分块 double buffer 搬入，调用 AscendC::Relu 完成计算后再搬出到 Global Memory。支持 (8,2048) 与 (2048,2048) 两种形状及 float32/float16 数据类型。
