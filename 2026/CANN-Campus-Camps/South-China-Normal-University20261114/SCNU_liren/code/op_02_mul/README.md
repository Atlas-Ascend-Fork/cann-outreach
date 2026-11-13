# op_02_mul

提交团队: 华南师范大学
提交者: 李俊杰


## 算子实现介绍

一、Tiling（Host）：由输入 shape 连乘求得 length=16384，tileNum=8，写入 MulTilingData；
   SetBlockDim(8) 保证 length 可被 8 核整除；ASCENDC_TPL_SEL_PARAM 按输入 dtype 选模板实例；
   InferShape 令 z 与 x 同形，InferDataType 令 z 与 x 同 dtype。
二、切分（Kernel）：blockLength = length / GetBlockNum() = 2048（每核）；
   tileLength = blockLength / tileNum / BUFFER_NUM = 128（每块）；Process 循环 16 次。
三、内存与对齐：三队列各 2 块 UB，每块 128 元素（fp32 512B、fp16 256B）；
   每次搬运量均为 32B 整数倍，核间偏移为 blockLength×核号，互不重叠。
四、流水：CopyIn（DataCopy 搬入 x、y）→ Compute（AscendC::Mul 逐元素相乘）
   → CopyOut（DataCopy 写回 z），入队/出队配合 FreeTensor 实现双缓冲复用。
