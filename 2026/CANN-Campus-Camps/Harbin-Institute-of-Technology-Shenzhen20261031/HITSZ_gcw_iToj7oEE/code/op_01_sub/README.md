# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: 陈欣彤


## 算子实现介绍

基于 Ascend C 实现向量减法算子，计算公式为 z = x - y。采用 8 个 NPU Core 并行计算，每个 Core 处理 2048 个元素，并划分为 16 个长度为 128 的数据块。使用 TPipe 和 TQue 管理输入输出缓冲区，通过 DataCopy 完成 Global Memory 与 Local Memory 之间的数据搬运，调用 AscendC::Sub 完成向量减法计算，最后将结果写回 Global Memory。根据输入数据类型分别实例化 half 和 float 核函数，支持 float16 和 float32。
