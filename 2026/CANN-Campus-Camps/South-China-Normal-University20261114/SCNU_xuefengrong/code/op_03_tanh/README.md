# op_03_tanh

提交团队: 华南师范大学
提交者: yeeee_91


## 算子实现介绍

anh 激活算子基于 Ascend C 在昇腾 AI Core 上原生实现，逐元素计算 y = tanh(x)，支持 float32/float16、ND 格式；多核并行切分，单核分块经搬入、计算、搬出流水完成，host 侧负责核数、分块与类型的 tiling 配置。

## 注意
