# op_03_tanh

提交团队: 华南师范大学
提交者: ZZLAtomGit


## 算子实现介绍

本算子采用单核实现，UB切分遵循32B对齐规则，Tiling策略将totalLength向上对齐至BLOCK_SIZE的整数倍，使用AscendC内置Tanh接口完成逐元素激活计算。
