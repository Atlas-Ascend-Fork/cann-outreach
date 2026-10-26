# expcustom

提交团队: 桂林理工大学
提交者: Feng_Li_Qiu


## 算子实现介绍

采用多核切分，按 blockFactor 将 totalNum 的连续区间分给各 AIV 核，尾核处理剩余元素。
Tiling策略：UB 分块按 32B 对齐，ubFactor 取 blockFactor、UB 预算和 MAX_UB_CHUNK 的最小值。
Kernel实现：CopyIn → Compute(Exp) → CopyOut，TQue 深度 2 双缓冲，非对齐回退 DataCopyPad。
