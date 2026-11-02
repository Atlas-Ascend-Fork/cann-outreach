# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: 何嘉俊


## 算子实现介绍

输入张量大小为 8 × 2048 = 16384 个元素，启动 8 个核，每个核处理 2048 个元素。每个核将自己的数据继续划分为多个 Tile：
- 每个 Tile 处理 128 个元素
- 使用 DataCopy 将输入从 Global Memory 搬到 Local Memory
- 两个输入分别放入 inX 和 inY 队列
从输入队列取出两个 Local Tensor，调用 AscendC 的：AscendC::Sub(zLocal, xLocal, yLocal, tileLength_);完成向量减法。将计算结果放入输出队列，再通过 DataCopy 搬回 Global Memory。Host 侧根据 dtype 选择模板实例。
