# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: 杨湘鹏


## 算子实现介绍

本算子基于 Ascend C Kernel 直调实现 z = x - y：8 核并行，每核 2048 元素，按 8×2 双缓冲切分为 128 元素 tile，通过 TPipe/TQue 完成 CopyIn → AscendC::Sub → CopyOut，支持 fp32/fp16。
