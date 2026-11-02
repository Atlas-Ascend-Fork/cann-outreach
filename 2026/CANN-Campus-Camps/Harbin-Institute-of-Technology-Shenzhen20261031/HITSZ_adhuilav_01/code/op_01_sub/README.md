# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: 朱炫翰


## 算子实现介绍

基于题目给出的 kernel_add.asc 参考实现完成 Ascend C 矢量减法 z = x - y。采用 8 核分片和队列搬入、计算、搬出，使用 AscendC::Sub，按输入 dtype 分派 float32 / float16 核函数。CANNJudge 提交 1036540，2/2 测试点 PASS，输出错误占比均为 0.00%。
