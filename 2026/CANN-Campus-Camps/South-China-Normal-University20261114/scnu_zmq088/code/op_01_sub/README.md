# op_01_sub

提交团队: 华南师范大学
提交者: zmq088


## 算子实现介绍

使用多核并行处理，单核内按 tileNum 分块。数据流采用 CopyIn-Compute-CopyOut 流水线设计，使用 TQue 队列管理 UB 内存，实现双缓冲。调用 AscendC::Sub 接口完成 float32/float16 数据类型的矢量减法（z = x - y）。
