# challenge07_sigmoidcustom

提交团队: 电子科技大学
提交者: Student_ZZX


## 算子实现介绍

基于 Ascend C 实现逐元素 Sigmoid。Tiling 将 8×2048 元素分配到 8 个 AI Vector Core，每核 16 个 tile 并采用双缓冲；float16 在 float32 中完成 Muls、Exp、Adds、Reciprocal，结合一次牛顿迭代提高精度，最后转换回输出类型。
