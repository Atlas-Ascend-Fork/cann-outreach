# expcustom

提交团队: 哈尔滨工业大学（深圳）
提交者: 杨湘鹏


## 算子实现介绍

基于 Ascend C 实现 Exp：Host 按 dtype 选择 tilingKey、8 核切分并计算 blockFactor/ubFactor；Kernel 使用 TPipe/TQue 双缓冲完成 CopyIn → AscendC::Exp → CopyOut，支持 fp32/fp16，输出 shape 与输入一致。
