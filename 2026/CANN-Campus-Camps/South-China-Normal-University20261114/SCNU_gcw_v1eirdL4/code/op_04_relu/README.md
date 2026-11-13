# op_04_relu

提交团队: 华南师范大学
提交者: gcw_v1eirdL4


## 算子实现介绍

基于 Ascend C 实现 ReLU(y=max(0,x))，支持 float32/float16、ND 格式。Tiling 侧按 AIV 核数均分总元素：blockFactor=CeilDiv(totalIdx,coreNum)，再按实际使用核数设置 BlockDim；UB 按「1 输入+1 输出、各双缓冲」共 4 块切分，ubFactor=FloorAlign(ubSize/(typeSize*4),32/typeSize)，即 32B 对齐并吃满 UB；按 dtype 设置 tilingKey 选择 half/float 模板实例。Kernel 侧用 GetBlockIdx 计算每核 Global Memory 起始地址与数据长度，TQue 实现双缓冲流水，搬运用 DataCopyPad 支持任意尾块长度，矢量计算长度向上对齐 32B，保证尾核不越界。
