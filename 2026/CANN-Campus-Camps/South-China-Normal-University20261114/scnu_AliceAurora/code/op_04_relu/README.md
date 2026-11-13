# op_04_relu

提交团队: 华南师范大学
提交者: AliceAurora


## 算子实现介绍

Relu 算子基于 Ascend C 开源仓工程实现，y = max(x, 0)，支持 float32 / float16，通过 tilingKey（schMode）区分 half 与 float 两个 kernel 实例。

Tiling 策略（op_host/relu_tiling.cpp）：
1. 由输入 shape 逐维相乘得到 totalNum（8×2048=16384 或 2048×2048=4194304），标量输入按 {1} 处理；
2. 按 dtype 取元素字节数（float=4、half=2），计算 32B 对齐的元素数 alignNum；
3. totalNum 小于阈值 1024 时退化为单核，避免空跑核；
4. 多核切分：blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum)，usedCoreNum = CeilDiv(totalNum, blockFactor)，SetBlockDim(usedCoreNum)；
5. UB 切分：ubFactor = FloorAlign(ubSize / (typeSize * 8), alignNum)，输入/输出队列各 BUFFER_NUM=2 块，约占 UB 一半，并限制不超过 blockFactor；
6. dtype 为 fp16/bf16 时 tilingKey 取 RELU_TPL_SCH_MODE_0（half 实例），fp32 时取 RELU_TPL_SCH_MODE_1（float 实例）。

Kernel 实现（op_kernel/relu.h）：
1. Init：按 GetBlockIdx()*blockFactor 计算本核在 GM 上的起始地址并 SetGlobalBuffer，同时对最后一个核裁剪尾块长度；初始化输入/输出队列（双缓冲）；
2. Process：按 ubFactor 分块循环，每轮处理 currentNum = min(ubLength_, 剩余长度)，所有偏移和长度均按 32B 对齐；
3. CopyIn/CopyOut：使用 DataCopy 在 GM 与 UB 之间搬运数据，队列 EnQue/DeQue 配对并及时 FreeTensor；
4. Compute：使用标量矢量指令 Maxs 一步完成 y = max(x, 0)。

InferShape：输出 y 的 shape 与输入 x 一致。
