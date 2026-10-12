# Sigmoid 算子

按输入 shape 和 dtype 计算多核 Tiling。每个核按 UB 容量分块，使用双缓冲队列搬入和搬出数据，并调用 `AscendC::Sigmoid<T, false>` 完成逐元素计算。支持 float16 和 float32 ND 张量；输出与输入形状及类型相同。

在 CANNJudge Challenge07 平台验证：提交 426729，Pass 2/2，两组输出误差均为 0.00%。
