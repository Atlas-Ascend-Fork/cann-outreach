# challenge07_sigmoidcustom

提交团队: 个人开发者
提交者: Code_demo111111


## 算子实现介绍

单核实现，SetBlockDim(1)，不切核。UB 切分按 32B 对齐，ubFactor 向下对齐到 alignNum。Tiling 策略：totalLength 向上对齐到 BLOCK_SIZE 整数倍，作为 totalNum 和 blockFactor。搬运使用 DataCopyPad 处理尾块，避免越界。计算采用 sigmoid(x) = 0.5 + 0.5 * tanh(x/2) 提升 f16 精度。Workspace 使用系统接口获取，不能为 0。
