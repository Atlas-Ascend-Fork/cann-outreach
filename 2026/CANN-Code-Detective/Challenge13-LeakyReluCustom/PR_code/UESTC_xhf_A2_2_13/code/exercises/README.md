# exercises

提交团队: 电子科技大学
提交者: xhf_A2_2


## 算子实现介绍

实现支持 float32 / float16 的逐元素 LeakyReLU。Tiling 读取 negative_slope，并按 32 字节对齐分核和切分 UB。核函数分别计算正、负半轴，尾块使用 DataCopyPad。CANNJudge 2/2 PASS。
