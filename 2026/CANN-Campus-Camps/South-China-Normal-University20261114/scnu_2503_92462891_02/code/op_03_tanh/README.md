# op_03_tanh

提交团队: 华南师范大学
提交者: 曾理


## 算子实现介绍

使用多核实现：host 侧按 AIV 核数将 totalNum 均分为 blockFactor，
SetBlockDim 设为实际用到的核数，kernel 按 GetBlockIdx 取本核分片。
UB 切分按 32B 对齐：ubFactor = FloorAlign(UB_SIZE / (sizeof(T) * 8), 32 / sizeof(T))，
输入/输出队列双缓冲，另用 4 个 VECCALC TBuf 存放中间结果（共 8 块）。
计算采用 Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))，
矢量计算长度向上对齐到 32B，尾块用 DataCopyPad 回拷，不越界写 GM。
Tiling 策略：按 dtype 选择 schMode(tilingKey)，MODE_0 = half，MODE_1 = float。
