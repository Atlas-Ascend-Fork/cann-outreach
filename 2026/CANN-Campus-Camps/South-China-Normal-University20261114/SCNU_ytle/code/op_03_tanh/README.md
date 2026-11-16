# op_03_tanh

提交团队: 华南师范大学
提交者: ytle


## 算子实现介绍

Host 侧按 shape 连乘得 totalNum，取平台 coreNum/ubSize 算出 blockFactor 与 ubFactor，
SetBlockDim(coreNum)，并按 dtype 映射 schMode 选择 half / float 模板实例。
Kernel 侧每核按 blockIdx×blockFactor 定位 GM 起点、尾块裁剪，按 ubFactor 分块循环，
双缓冲三段式流水（CopyIn→Compute→CopyOut），搬运 32B 对齐。
tanh 不用内置 API，由 Muls→Exp→Exp→Sub→Add→Div 六个小算子拼接；
中间量放 UB 的 TBuf 复用；fp16 先 Cast 到 fp32 计算、末尾一次转回，并夹取 ±8 防溢出。
