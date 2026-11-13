# op_01_sub

提交团队: 华南师范大学
提交者: 周雨瑄


## 算子实现介绍

使用单核实现，UB 切分按 32B 对齐
Tiling 策略：totalLength 对齐到 BLOCK_SIZE 的整数倍。BLOCK_SIZE 为 32 字节；fp16 (half) 单元素 2B，对齐粒度为 16 个元素；fp32 (float) 单元素 4B，对齐粒度为 8 个元素。Host 侧先将原始总长度向上对齐至对应粒度；单核内部采用双 Buffer 分块，保证每一块 UB Buffer 的字节大小为 32B 整数倍；核内循环执行搬入、减法计算（z = x - y）、结果搬出。
