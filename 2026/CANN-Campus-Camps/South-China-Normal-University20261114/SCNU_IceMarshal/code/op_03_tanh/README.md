# op_03_tanh

提交团队: 华南师范大学
提交者: 黄明沐


## 算子实现介绍

使用多核实现，核间按blockFactor切分，UB循环按ubFactor处理，搬运按32B对齐

计算过程：tanh(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x))，由Exp/Muls/Sub/Add/Div小算子拼接；float16先升精度到float计算，再CAST_RINT转回

Tiling策略：totalNum按coreNum均分并向上对齐到BLOCK_SIZE（32B）的整数倍；ubFactor按UB容量（入/出队列双缓冲+float临时buffer）向下对齐32B
