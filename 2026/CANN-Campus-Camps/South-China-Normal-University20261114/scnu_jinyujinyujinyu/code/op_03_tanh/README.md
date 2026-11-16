# op_03_tanh

提交团队: 华南师范大学
提交者: gcw_DKh6obTX


## 算子实现介绍

使Ascend C 实现逐元素 Tanh 算子，支持形状为 (8, 2048) 的 FP16 和 FP32 张量。Host 侧将数据分给 8 个核，每个核处理 2048 个元素，每次搬运和计算 128 个元素。Kernel 侧使用双缓冲，通过 Abs、Exp、Div 等基础向量运算计算 Tanh 的绝对值。
