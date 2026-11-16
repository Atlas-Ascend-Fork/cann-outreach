# op_02_mul

提交团队: 华南师范大学
提交者: yeeee_91


## 算子实现介绍

Mul（矢量逐元素乘法）算子基于 Ascend C 在昇腾 AI Core 上原生实现，对两个同形状、同类型输入张量按位相乘，输出 z = x * y，支持 float32 /float16、ND 格式；多核并行切分数据，单核内分块经 CopyIn、Compute、CopyOut 三级流水完成，host 侧负责 shape、dtype 与核数的 tiling 配置。
