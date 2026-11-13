# op_02_mul

提交团队: 华南师范大学
提交者: AliceAurora


## 算子实现介绍

本算子使用 Ascend C 实现逐元素乘法 z = x * y，支持 float32 / float16，通过 tiling key 模板参数（ASCENDC_TPL_DATATYPE_DECL）区分两种数据类型。

Tiling 策略（op_host/mul.cpp）：
1. 由输入 Tensor 获取元素总数 totalLength = 8 × 2048 = 16384；
2. 按 8 个 AIV 核均分，每核处理 blockLength = totalLength / GetBlockNum() = 2048 个元素；
3. 每核再按 TILE_NUM = 8 分块，配合 BUFFER_NUM = 2 双缓冲，每块 tileLength = 128 个元素；
4. 结果写入 MulTilingData{totalLength, tileNum}，SetBlockDim(8)，workspace 设为 0；
5. dtype 通过 ASCENDC_TPL_SEL_PARAM 设置 tiling key，使 kernel 按 float32 / float16 分别实例化。

UB 切分与对齐：每块 128 个元素，float32 为 512 Byte、float16 为 256 Byte，均按 32B（BLOCK_SIZE）对齐，满足 DataCopy 的对齐要求。

Kernel 实现（op_kernel/mul.cpp）：
1. Init：按 AscendC::GetBlockIdx() 计算本核 GM 偏移并 SetGlobalBuffer；TPipe 为 x、y、z 各申请 2 块队列（VECIN/VECIN/VECOUT）；
2. Process：循环 tileNum × BUFFER_NUM = 16 次，执行 CopyIn → Compute → CopyOut 流水；
3. CopyIn：两次 DataCopy 把 x、y 分块搬入 UB 并 EnQue；
4. Compute：DeQue 后调用 AscendC::Mul 做逐元素乘，结果 EnQue 到 VECOUT 并释放输入；
5. CopyOut：DeQue 结果后 DataCopy 写回 z 的 GM 并释放 UB。

InferShape / InferDataType：输出 z 的 shape 与 dtype 均与输入 x 一致。
