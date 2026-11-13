# op_02_mul

提交团队: 华南师范大学
提交者: 曾理


## 算子实现介绍

本算子基于 Ascend C 自定义算子工程实现逐元素乘法 z = x * y，支持 float32 / float16、ND 格式，输出 shape 与 dtype 与 x 一致；模板参数 DT_X 通过 tiling key 在 fp32 / fp16 两个实例间选择。

多核切分：host 侧读取 GetCoreNumAiv() 并 SetBlockDim 满核启动，只把 GetShapeSize() 写进 tiling 的 length 字段；kernel 侧用 GetBlockNum()/GetBlockIdx() 现算 perCore = ceil(length / blockNum)，再向上对齐到 ALIGN_ELEM = 32 / sizeof(DT_X)（fp32 8 元素、fp16 16 元素），使每核 GM 起始偏移都是 32B 整数倍；越界核 blockLength 置 0 空转，末核按剩余元素截断。

UB 切分：每块固定 TILE_BYTES = 8192，tileLength = 8192 / sizeof(DT_X)（fp32 2048、fp16 4096 个元素）；x、y 各一个 VECIN 队列、z 一个 VECOUT，BUFFER_NUM = 2 做 Double Buffer，共 3×2×8KB = 48KB；按 CopyIn → Compute → CopyOut 三段流水，靠 EnQue/DeQue 让搬运与计算重叠。

尾块与对齐：CopyIn/CopyOut 统一用 DataCopyPad，blockLen 传实际字节数，尾块不必 32B 对齐也不越界；Compute 中 calcNum 向上对齐 32B 后再调 AscendC::Mul，多算的元素留在 tile buffer 内，回拷只写有效的 currentNum 个。

验证：验证集 2/2 PASS，错误占比 0.00%，用时 16.17 μs / 18.28 μs（最优 4.06 / 3.75 μs）。
