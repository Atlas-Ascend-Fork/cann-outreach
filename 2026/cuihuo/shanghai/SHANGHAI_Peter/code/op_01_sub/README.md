# op_01_sub

提交团队: 上海淬火
提交者: peter


## 算子实现介绍

Ascend C 向量减法算子(z = x − y):模板化支持 fp16/fp32,多核 + 单核 tile 两级切分，经 TQue 队列驱动 GM↔UB 搬运与 Vector 计算的流水化执行。
