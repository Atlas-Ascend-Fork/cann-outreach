# challenge07_sigmoidcustom

提交团队: 电子科技大学
提交者: UESTC_xhf


## 算子实现介绍

基于 8 个 AI Vector Core 并行处理 8×2048 输入，每核采用 16 个逻辑 tile 和双缓冲流水。float16 输入先转换为 float 计算，通过 Muls、Mins、Exp、Adds、Reciprocal 实现 Sigmoid，并使用一次牛顿迭代提升倒数精度，最后转换回原始数据类型。
