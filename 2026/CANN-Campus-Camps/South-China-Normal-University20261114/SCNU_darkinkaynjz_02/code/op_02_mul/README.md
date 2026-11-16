# op_02_mul

提交团队: 华南师范大学
提交者: darkinkaynjz


## 算子实现介绍

使用 Ascend C 实现向量乘法，支持 float16 和 float32。Host 根据数据类型选择模板实例，向量核以 128 个元素为一块分配计算。代码由 Codex 辅助实现。
