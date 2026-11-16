# op_04_relu

提交团队: 华南师范大学
提交者: 2503_92462891


## 算子实现介绍

使用多核实现：host 侧按 AIV 核数将 totalNum 均分为 blockFactor，SetBlockDim 设为实际用到的核数；kernel 按 GetBlockIdx 计算本核 GM 起始地址，尾核取剩余长度，避免越界。
UB 切分按 32B 对齐：ubFactor = FloorAlign(UB_SIZE / (sizeof(T) * 4), 32 / sizeof(T))；输入/输出各 2 块组成 Double Buffer（共 4 块），CopyIn / Compute / CopyOut 三段流水。
计算执行 y = max(0, x)，直接调用 AscendC::Relu；矢量计算长度向上对齐到 32B，尾块用 DataCopyPad 只回拷有效元素。
Tiling 策略：按 dtype 选择 schMode(tilingKey)，MODE_0 → half，MODE_1 → float。
