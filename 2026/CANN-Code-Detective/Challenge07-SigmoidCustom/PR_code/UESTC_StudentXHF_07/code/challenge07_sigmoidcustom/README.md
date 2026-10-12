# challenge07_sigmoidcustom

提交团队: 电子科技大学
提交者: StudentXHF


## 算子实现介绍

基于 8 个 AI Vector Core 并行处理 8×2048 输入，每核按 32B 对齐进行双缓冲切分。float16 先转换为 float 计算；通过 Muls、Exp、Adds、Reciprocal 组合实现 Sigmoid，并用一次牛顿迭代提升倒数精度，最后转换回原始 dtype。
