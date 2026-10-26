# expcustom

提交团队: 湖北工业大学
提交者: muluzhe


## 算子实现介绍

使用 AscendC 实现 Exp（y = e^x）逐元素算子，支持 float32 / float16，采用多核切分 + 单块 UB 的
CopyIn → Compute → CopyOut 流程，任意 shape（含非 32B 对齐尾块）均正确。
Tiling 策略：
1) shape 各维相乘得到 totalLength；按“每核至少 2048 元素”确定分核数（(8,2048) → 8 核），
   分核数不超过平台 AIV 核数；
2) blockFactor = ceil(totalLength / 分核数)，并向上对齐到 32B（float 8 元素 / float16 16 元素），
   保证每个核在 GM 上的起始地址满足 DataCopy 对齐要求；
3) ubFactor = min(UB 可用元素数, blockFactor)，再向下对齐到 32B，本算例每核一次搬完、无内层循环；
4) 按输入 dtype 设置 tilingKey（0 = float16，1 = float32），workspace 置 0。
   分核阈值 2048 由本机 4/8/12/16/20/40 核扫描确定（每核 2048 元素时最优）。
Kernel 实现：
1) 本核 GM 偏移 = blockFactor × GetBlockIdx()，尾核按剩余元素截断；
2) 一块 TBuf 划分为输入/输出两半，输出区偏移单独按 32B 向上对齐（与拷贝长度解耦，避免尾核
   blockLength 非对齐导致 UB 地址不对齐）；
3) DataCopy(GM→UB) → AscendC::Exp → DataCopy(UB→GM)，用 SetFlag/WaitFlag 完成
   MTE2→V→MTE3 同步；
4) 对长度非 32B 对齐的尾块改用 DataCopyPad 按字节搬移，保证任意 shape/偏移正确。
