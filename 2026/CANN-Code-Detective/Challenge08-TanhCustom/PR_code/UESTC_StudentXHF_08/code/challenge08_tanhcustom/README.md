# challenge08_tanhcustom

提交团队: 电子科技大学
提交者: StudentXHF


## 算子实现介绍

使用多核切分与双缓冲；Host 根据 totalNum、AIV 核数和 UB 容量计算 blockFactor/ubFactor；Kernel 以 1-2/(exp(2x)+1) 组合 Muls、Exp、Adds、Div 实现 Tanh，支持 float16/float32。
