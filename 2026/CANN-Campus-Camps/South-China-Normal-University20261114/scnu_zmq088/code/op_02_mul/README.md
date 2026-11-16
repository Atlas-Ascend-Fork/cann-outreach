# op_02_mul

提交团队: 华南师范大学
提交者: zmq088


## 算子实现介绍

使用多核并行处理，Host侧根据总数据长度配置BlockDim，单核内根据tileNum将数据分块。
Kernel侧数据流采用 CopyIn-Compute-CopyOut 流水线设计，使用 TQue 队列管理 UB 内存，并开启双缓冲（BUFFER_NUM=2）提升并发性能。
计算阶段调用 AscendC::Mul 接口，完成 float32/float16 数据类型的逐元素矢量乘法（z = x * y）。
