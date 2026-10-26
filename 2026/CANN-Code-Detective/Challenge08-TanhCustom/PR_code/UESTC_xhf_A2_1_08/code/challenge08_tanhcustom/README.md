# challenge08_tanhcustom

提交团队: 电子科技大学
提交者: 徐海峰

## 算子实现介绍

使用多核切分和双缓冲处理（8、2048）输入；Host 侧根据 AIV 核数与 UB 容量计算 blockFactor 和 ubFactor，并按 float16/float32 选择 tiling key；Kernel 以 `1 - 2 / (exp(2x) + 1)` 组合 Muls、Exp、Adds、Duplicate、Div 实现逐元素 Tanh，输出 shape 与输入一致。
