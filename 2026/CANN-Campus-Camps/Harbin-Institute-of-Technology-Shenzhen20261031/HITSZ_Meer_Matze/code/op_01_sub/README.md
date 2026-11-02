# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: Meer_Matze


## 算子实现介绍

采用 8 核并行实现，每个核负责处理 2048 个元素。每核将数据进一步划分为 `tileNum × BUFFER_NUM` 个 Tile，使用双缓冲队列管理输入和输出数据，通过 DataCopy 完成 Global Memory 与 UB 之间的数据搬运，并调用 `AscendC::Sub` 执行矢量减法。

UB 切分： 每个 Tile 包含 128 个元素，FP16 和 FP32 均满足 32B 对齐要求。

Tiling 策略： 根据 `totalLength` 和核数计算 `blockLength`，再结合 `tileNum` 和 `BUFFER_NUM` 确定 `tileLength`。本题固定输入规模为 \\(8\times2048\\)，由 8 个核均匀处理。
