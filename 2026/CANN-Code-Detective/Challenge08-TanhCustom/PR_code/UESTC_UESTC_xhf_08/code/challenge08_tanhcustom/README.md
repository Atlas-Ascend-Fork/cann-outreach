# challenge08_tanhcustom

提交团队: 电子科技大学
提交者: UESTC_xhf


## 算子实现介绍

使用 Vector Core 多核切分与 Double Buffer；支持 float16/float32。通过 Muls、Exp、Adds、Duplicate、Div 实现 tanh(x)=1-2/(exp(2x)+1)，并按 UB 容量进行 32B 对齐分块。
