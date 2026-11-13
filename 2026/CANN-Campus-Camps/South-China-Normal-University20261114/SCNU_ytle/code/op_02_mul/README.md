# op_02_mul

提交团队: 华南师范大学
提交者: 陈锭凯


## 算子实现介绍

aWQfBxGN2FW-U7346R8fcJJy8核并行 + 核内二次分块 + 双缓冲。Host 侧从输入 shape 连乘得到 length=16384，
tileNum 固定 8，SetBlockDim(8)，并按 dtype 经 ASCENDC_TPL_SEL_PARAM 选择
float / half 模板实例；InferShape、InferDataType 令输出 z 与输入 x 同形同 dtype。
核内 blockLength = length/8 = 2048，tileLength = 2048/8/2 = 128，循环 16 次；
三个队列各分配 2 块 UB（fp32 512B、fp16 256B，均满足 32B 对齐），
流水为 DataCopy 搬入 → AscendC::Mul → DataCopy 搬出，双缓冲复用。
