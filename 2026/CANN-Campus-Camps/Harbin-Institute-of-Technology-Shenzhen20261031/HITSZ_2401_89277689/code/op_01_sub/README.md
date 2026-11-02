# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: 2401_89277689


## 算子实现介绍

采用多核向量化实现，默认使用 8 核，每核处理一段连续等长数据，实现 z = x - y。UB 中直接划分 x/y/z 三个 LocalTensor，各缓冲区及地址偏移按 32B 对齐；通过 DataCopy 搬入、Sub 计算、DataCopy 搬出，并使用 SetFlag/WaitFlag 显式同步。Tiling 策略：totalLength 对齐到 BLOCK_SIZE 的整数倍，按 1/2/4/8/16/32 核均分，保证无尾块、各核输出不重叠。
