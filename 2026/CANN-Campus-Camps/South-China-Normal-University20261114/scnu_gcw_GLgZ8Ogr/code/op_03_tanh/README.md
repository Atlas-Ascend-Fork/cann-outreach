# op_03_tanh

提交团队: 华南师范大学
提交者: 吴靖飞


## 算子实现介绍

Tanh 采用多核 + UB 分块的 Ascend C 实现。Host 侧 Tiling 根据输入 shape 得到 totalNum，按 32B 对齐将数据切分到各核（blockFactor），并按 UB 容量切分 ubFactor；Kernel 侧每核处理 blockFactor 个元素，核内按 ubFactor 循环执行 CopyIn→Compute→CopyOut，输入/输出队列各双缓冲，另用一块临时 buffer。计算按 (e^x − e^(−x)) / (e^x + e^(−x)) 实现，依次调用 Exp、Muls、Sub、Add、Div，按 dtype 选择 half/float 模板实例。
