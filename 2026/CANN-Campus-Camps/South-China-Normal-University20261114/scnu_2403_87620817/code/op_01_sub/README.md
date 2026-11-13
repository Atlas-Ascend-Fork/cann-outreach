# op_01_sub

提交团队: 华南师范大学
提交者: 2403_87620817


## 算子实现介绍

采用 Ascend C 向量计算范式实现减法算子，计算公式为 z = x - y。

采用 8 个 AI Core 进行多核并行，每个核独立处理一段数据。单核内部采用 Tile 分块策略，并通过 TPipe 和 TQue 管理 UB 内存及数据队列，完成 CopyIn、Compute、CopyOut 三个阶段的数据处理。

Tiling 策略：totalLength 按 AI Core 数量、tileNum 和 BUFFER_NUM 进行均匀切分，确保每个 Tile 的数据长度满足 32B 对齐要求。对于输入 Shape=(8, 2048)，每核处理 2048 个元素，每个 Tile 处理 128 个元素。

支持 FP32 和 FP16 数据类型，使用 AscendC::Sub API 执行逐元素向量减法，最终将计算结果写回 Global Memory。
