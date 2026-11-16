# op_03_tanh

提交团队: 华南师范大学
提交者: darkinkaynjz


## 算子实现介绍

使用基础向量运算组合双曲正切，支持 float16 和 float32。以 float32 计算指数和比值，近零区间保留输入以减小抵消误差。代码由 Codex 辅助实现。
