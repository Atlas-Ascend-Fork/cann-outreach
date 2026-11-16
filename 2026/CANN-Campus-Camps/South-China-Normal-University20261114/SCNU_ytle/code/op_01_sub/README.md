# op_01_sub

提交团队: 华南师范大学
提交者: ytle


## 算子实现介绍

8核并行 + 核内二次分块 + 双缓冲。blockLength=totalLength/8=2048，
tileLength=2048/8/2=128，循环16次；每核按blockLength*GetBlockIdx()定位GM起点。
UB每队列2块（fp32 512B、fp16 256B，均32B对齐），
流水为 DataCopy → AscendC::Sub → DataCopy。核函数模板化，按dtype分派 float/half。
