# op_02_mul

提交团队: 华南师范大学(石牌校区)
提交者: hachijiang


## 算子实现介绍

1. Host：TilingFunc 取 totalLength=16384、tileNum=8、SetBlockDim(8)，并补 InferShape/InferDataType（输出继承 x）。
2. Kernel：8 核均分（每核 2048），核内 tileNum=8 + 双缓冲切成 128 元素/块，16 轮 CopyIn→Compute→CopyOut。
3. 搬运用 TQue + DataCopy，计算用 AscendC::Mul，tiling key 按 dtype 选择 KernelMul&lt;float/half&gt;。
