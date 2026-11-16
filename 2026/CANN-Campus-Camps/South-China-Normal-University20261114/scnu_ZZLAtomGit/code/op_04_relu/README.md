# op_04_relu

提交团队: 华南师范大学
提交者: ZZLAtomGit


## 算子实现介绍

本算子采用多核并行实现，UB 切分遵循 32B 对齐规则，Tiling 策略将单核分配长度对齐至 BLOCK_SIZE 的整数倍。
