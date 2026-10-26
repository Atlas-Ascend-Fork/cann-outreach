# expcustom

提交团队: 电子科技大学
提交者: Student_ZZX


## 算子实现介绍

基于 Ascend C 实现 Exp 自然指数算子，支持 float16/float32；Host 侧按核数、数据类型和 UB 容量动态切分，Kernel 侧完成分块搬运与向量化 Exp 计算。
