# op_04_relu

提交团队: 华南师范大学
提交者: yeeee_91


## 算子实现介绍

本题基于 Ascend C 在昇腾 NPU 上实现 ReLU 算子 y=max(0,x)：补全 tiling 的切片数、启动核数与 UB 预算，以及 kernel 的多核 GM 切分、32B 对齐双缓冲搬运和 Maxs计算，支持 fp32/fp16，形状 (8,2048)/(2048,2048)。
