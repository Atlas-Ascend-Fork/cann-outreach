# challenge11_sqrtcustom

提交团队: 易思博
提交者: scanz


## 算子实现介绍

Ascend C 逐元素平方根算子：host 侧按输入总元素数与 AIV 核数进行 256 元素对齐切分，并按 float16/float32 选择 tiling key；kernel 侧通过双缓冲队列完成 CopyIn、Sqrt 和 CopyOut，处理末尾分块。CANNJudge 4/4 PASS，四项输出错误占比均为 0.00%。在 Ascend910B3、CANN 9.0.0 容器中 host 与 kernel 编译通过，CPU 仿真 16384 元素验证通过。代码由 Codex 辅助实现与验证。
