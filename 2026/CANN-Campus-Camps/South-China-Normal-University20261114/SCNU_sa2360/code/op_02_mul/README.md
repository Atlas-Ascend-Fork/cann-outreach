# op_02_mul

提交团队: 华南师范大学
提交者: sa2360


## 算子实现介绍

Ascend C 矢量乘法采用多核并行处理，各核以轮转方式处理连续数据 tile，将 x、y 搬入 UB 后调用 AscendC::Mul 计算并写回 z。每个 tile 为 256 个元素，float32 占 1024B、float16 占 512B，满足 32B 对齐。Host 侧传入 totalLength，核函数按 block 数和 tile 大小分配工作；本题长度 16384 是 BLOCK_SIZE=256 的整数倍，无尾块，同时实现了边界长度保护。支持 float32 和 float16。
