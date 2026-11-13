# exercises

提交团队: 智源研究院
提交者: taojian


## 算子实现介绍

基于 Ascend C 实现 LeakyReLU。Host 侧按 FP32 / FP16 选择 tilingKey，并将 negative_slope 写入 tilingData。Kernel 按核和 UB 分块，通过 CopyIn、Compute、CopyOut 完成逐元素计算。FP16 输入在 FP32 中计算，最后转回 FP16。

CANNJudge 提交 #1257006：2/2 个测试点通过，输出错误占比均为 0.00%。
