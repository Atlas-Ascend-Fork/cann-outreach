# expcustom

提交团队: 梧桐森智能科技（东莞）有限公司
提交者: sws168


## 算子实现介绍

基于 Ascend C 实作逐元素 Exp，支援 float32 与 float16，输出形状与输入一致。Tiling 依 AIV 核数及 UB 容量计算 blockFactor、ubFactor，并按 UB block 对齐；Kernel 通过 TQue 缓冲分块搬入数据、调用 AscendC::Exp 计算并搬出结果，涵盖尾块。经 CANNJudge 验证，2/2 测试点通过。
