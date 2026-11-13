# challenge12_abscustom

提交团队: 电子科技大学
提交者: Student_ZZX


## 算子实现介绍

基于 Ascend C 实现逐元素 Abs：按输入 dtype 选择 float16/float32 核函数，以 32B 对齐的多核分块和双缓冲 UB 分批搬运，调用 AscendC::Abs，并处理尾块。CANNJudge 验证集通过 2/2。
