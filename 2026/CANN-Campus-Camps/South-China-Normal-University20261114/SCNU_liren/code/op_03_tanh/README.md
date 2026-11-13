# op_03_tanh

提交团队: 华南师范大学
提交者: 李俊杰


## 算子实现介绍

Host 侧：TilingFunc 由输入 shape 连乘得 totalNum=16384，按平台 coreNum / ubSize 计算
blockFactor（每核元素数）与 ubFactor（每块元素数），SetBlockDim(coreNum)；再按 dtype
映射 schMode（fp16→0 走 half 实例，fp32→1 走 float 实例）选择核函数模板。
Kernel 侧：每核按 blockIdx×blockFactor 定位 GM 起点并裁剪尾块，按 ubFactor 分块循环，
走 CopyIn→Compute→CopyOut 三段式流水，队列 2 块 buffer 实现双缓冲，搬运量 32B 对齐。
计算不使用 tanh API，由小算子拼接：Muls(-x) → Exp(e^x) → Exp(e^-x) → Sub(分子)
→ Add(分母) → Div；中间量放在 UB 的 TBuf 中复用；fp16 输入先 Cast 到 fp32 计算、
末尾一次 Cast 回 fp16，并在计算前把 x 夹到 [-8,8] 防止 exp 溢出。
