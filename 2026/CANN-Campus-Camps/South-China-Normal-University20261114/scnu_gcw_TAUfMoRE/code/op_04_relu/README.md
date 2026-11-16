# op_04_relu

提交团队: 华南师范大学
提交者: 马晓北


## 算子实现介绍

基于 Ascend C 编程语言开发。Host 侧进行 Tiling 策略切分并配置多核（BlockDim）并行；Device 侧 Kernel 将数据从 Global Memory 搬运至 UB（Unified Buffer），利用向量指令 Maxs 实现高效的逐元素计算，并结合双缓冲（DoubleBuffer）机制实现流水线并行，最大化昇腾 NPU 算力。
