# op_02_mul

提交团队: 华南师范大学
提交者: ZZLAtomGit


## 算子实现介绍

本算子采用单核实现，UB切分遵循32B对齐规则，Tiling策略将总长度totalLength对齐至BLOCK_SIZE的整数倍。
