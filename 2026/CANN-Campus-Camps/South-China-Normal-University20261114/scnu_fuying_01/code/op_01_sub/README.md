# op_01_sub

提交团队: 华南师范大学
提交者: gcw_vuifW5gj


## 算子实现介绍

单核实现，UB 切分严格按照 **32B** 对齐；
Tiling 策略：将总长度 `totalLength` 向上对齐到 `BLOCK_SIZE(32B)` 的整数倍，满足 DataCopy 硬件搬运对齐要求；
多核并行拆分任务，单核内部再分 Tile 循环执行
支持fp16(half)、fp32(float)两种数据类型。
