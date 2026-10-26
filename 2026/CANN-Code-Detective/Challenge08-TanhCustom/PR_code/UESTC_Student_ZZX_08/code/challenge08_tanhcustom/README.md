# challenge08_tanhcustom

提交团队: 电子科技大学
提交者: Student_ZZX


## 算子实现介绍

基于 Ascend C 实现逐元素 Tanh。将 8×2048 元素分配到 8 个 AI Vector Core，每核 16 个 tile、双缓冲；在 float32 中按 2/(1+exp(-2x))-1 计算，裁剪指数参数以避免溢出，并对倒数做一次牛顿迭代。float16 输入在 float32 中计算后转换回输出类型。
