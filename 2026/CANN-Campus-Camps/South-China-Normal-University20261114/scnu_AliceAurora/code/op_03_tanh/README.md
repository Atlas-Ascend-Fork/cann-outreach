# op_03_tanh

提交团队: 华南师范大学
提交者: AliceAurora


## 算子实现介绍

Tanh 算子基于 Ascend C 开源仓工程实现，y = tanh(x)，支持 float32 / float16，通过 tilingKey（schMode）区分 half 与 float 两个 kernel 实例。

Tiling 策略（op_host/tanh_tiling.cpp）：
1. 由输入 shape 逐维相乘得到 totalNum（16384），标量输入按 {1} 处理；
2. 按 dtype 取元素字节数（float=4、half=2），计算 32B 对齐的元素数 alignNum；
3. 多核切分：blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum)，usedCoreNum = CeilDiv(totalNum, blockFactor)，SetBlockDim(usedCoreNum)；
4. UB 切分：ubFactor = FloorAlign(ubSize / (typeSize * 4), alignNum)，为双缓冲和计算中间结果预留空间，并限制不超过 blockFactor；
5. dtype 为 fp16/bf16 时 tilingKey 取 TANH_TPL_SCH_MODE_0（half 实例），fp32 时取 TANH_TPL_SCH_MODE_1（float 实例）。

Kernel 实现（op_kernel/tanh.h）：
1. Init：按 GetBlockIdx()*blockFactor 计算本核在 GM 上的偏移，裁剪最后一个核的尾块长度；初始化输入/输出队列（各 BUFFER_NUM=2）与两个 float 类型的 TBuf 临时缓冲；
2. Process：按 ubFactor 分块循环，每轮处理 currentNum = min(ubLength_, 剩余长度)，保证搬运长度 32B 对齐；
3. CopyIn/CopyOut：使用 DataCopy 在 GM 与 UB 之间搬运数据，队列 EnQue/DeQue 配对并及时 FreeTensor；
4. Compute：tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)。float 输入直接计算；half 输入先用 Cast(CAST_NONE) 转 float 计算，避免 exp(2x) 溢出并保证精度，最后 Cast(CAST_RINT) 转回 half。

InferShape：输出 y 的 shape 与输入 x 一致。
