# op_04_relu

提交团队: 华南师范大学
提交者: 吴靖飞


## 算子实现介绍

多核切分：Tiling 从输入 shape 得到 totalNum（8×2048=16384），blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), 32B元素数)，按实际用到的核数 SetBlockDim；每个核处理 [blockIdx × blockFactor, …) 区间，最后一个核按剩余元素数处理，长度仍保证 32B 对齐。
UB 切分：输入队列 2 块 + 输出队列 2 块（DoubleBuffer），ubFactor 按 UB 一半、4 块 buffer 计算并 32B 对齐，且不超过 blockFactor；核内按 ubFactor 分 tile，tile 长度与 GM 偏移均为 32B 整数倍，无需 DataCopyPad。
流水：TPipe 分配队列，CopyIn 搬入 → Compute 调 Maxs(dst, src, 0, num) 做 ReLU → CopyOut 搬出，EnQue/DeQue 实现双缓冲重叠。
Tiling 参数：ReluTilingData { totalNum, blockFactor, ubFactor } 按值传入核函数；host 侧按 dtype 设置 tilingKey，fp16 走 Relu&lt;half&gt;、fp32 走 Relu&lt;float&gt;。
典型配置（8×2048，fp32）：64 核时 blockFactor=256、ubFactor=256，每核 1 次循环；48 核时 blockFactor=344（末核 216）。
