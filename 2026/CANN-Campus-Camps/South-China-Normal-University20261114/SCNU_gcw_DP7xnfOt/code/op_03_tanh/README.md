# op_03_tanh

提交团队: 华南师范大学
提交者: 何建基


## 算子实现介绍

使用 Ascend C 实现 Tanh 激活算子 y=tanh(x)。Host 侧按输入 shape 计算总元素数，根据 dtype 计算 32B 对齐的 blockFactor 与 ubFactor，并按实际使用核数设置 blockDim；Kernel 侧 multi-core 切分后按 UB 分块 double buffer 搬入，调用 AscendC::Tanh 完成计算后再搬出到 Global Memory。支持 float32 与 float16。
