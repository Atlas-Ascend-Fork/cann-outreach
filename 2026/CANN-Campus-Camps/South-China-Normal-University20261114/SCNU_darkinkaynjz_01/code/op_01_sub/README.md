# op_01_sub

提交团队: 华南师范大学
提交者: darkinkaynjz


## 算子实现介绍

使用 Ascend C 实现向量减法，支持 float16 和 float32。按 128 个元素分块，在可用向量核上分配数据，使用双缓冲队列。代码由 Codex 辅助实现并通过 CANNJudge 验证。
