# op_02_mul

提交团队: 华南师范大学
提交者: gcw_pt8ZEjCI


## 算子实现介绍

本算子实现逐元素乘法 z=x*y，支持 float32 和 float16。Tiling 策略上启动 8 个核进行多核并行，单核处理 2048 个元素，数据长度完美满足 32B 对齐要求，无尾块开销。在 UB 切分上，将 TILE_NUM 设为 1 以避免小块碎片化，并开启双缓冲（BUFFER_NUM=2）。Kernel 侧仅执行两轮流水，通过 Ping-Pong 机制高效重叠 DataCopy 搬运与 AscendC::Mul 计算，在保证不整除兼容性的同时，最大化降低了小 Shape 场景下的调度耗时。
