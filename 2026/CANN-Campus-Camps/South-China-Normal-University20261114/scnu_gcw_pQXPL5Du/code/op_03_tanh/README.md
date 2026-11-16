# op_03_tanh

提交团队: 华南师范大学
提交者: gcw_pQXPL5Du


## 算子实现介绍

本算子使用 Ascend C，基于公式实现 Tanh 逐元素运算，通过 tiling 做多核并行，搭配双缓冲流水线在 UB 上分块计算，保证精度同时适配昇腾 AICore。
