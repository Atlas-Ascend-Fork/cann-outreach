# expcustom

提交团队: 个人开发者
提交者: Code_demo111111


## 算子实现介绍

基于 Ascend C 实现，支持 float32/float16。Host 侧进行多核切分与 UB 切分，长度均按 32B 对齐；Kernel 侧通过 CopyIn-Compute-CopyOut 流水线，使用 AscendC::Exp 指令完成逐元素指数计算，并管理 TPipe/TQue 实现高效的数据搬运。
