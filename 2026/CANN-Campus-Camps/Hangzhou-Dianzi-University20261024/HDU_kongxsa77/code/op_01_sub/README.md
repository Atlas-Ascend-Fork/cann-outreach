# op_01_sub

提交团队: 杭州电子科技大学
提交者: kongxsa77


## 算子实现介绍

Ascend C 原生直调实现 z = x − y。单核启动，TPipe + TQue（深度 1）管理 UB，
每块 4096 元素：DataCopy 搬入 → AscendC::Sub → DataCopy 写回。
totalLength 由 shape 推出、按 32B 对齐切块，尾部标量补齐；
按 dtype 分派 float32/float16。UB 峰值约 48KB。
