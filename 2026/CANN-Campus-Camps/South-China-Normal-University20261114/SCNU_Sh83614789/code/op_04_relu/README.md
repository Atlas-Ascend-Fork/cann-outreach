# op_04_relu

提交团队: 华南师范大学
提交者: Sh83614789


## 算子实现介绍

**算子实现介绍（ReLU）**

本算子基于 Ascend C 实现逐元素 ReLU（y = max (0, x)），支持 float32/float16、ND 排布，代码按开源仓规范分 op_host 与 op_kernel 两部分。

**多核切分**：host 取 AIV 核数，按 blockFactor = CeilDiv (totalNum, coreNum) 均分，实际核数由 CeilDiv (totalNum, blockFactor) 得出并 SetBlockDim 下发；以 (8,2048) 的 16384 个元素为例，8 核各处理 2048 个。kernel 用 GetBlockIdx 定位本核 GM 起始偏移，处理长度取 min (blockFactor, totalNum - blockFactor * blockIdx)，尾核自动收敛不越界。

**切块与流水**：ubFactor 按 UB 容量与 4 块 tensor（双缓冲队列加缓冲）推导，并对齐到 32 字节。两个 TQue 各开 2 个缓冲构成输入输出双缓冲，搬运用 DataCopyPad 支持非对齐长度，尾块不越界；Process 按 ubLength 循环 CopyIn → Compute → CopyOut，使 MTE2 搬运、Vector 计算、MTE3 回写三者流水重叠。

**计算与精度**：Compute 出队后仅调用一条 Maxs (y, x, 0, calcCount) 完成逐元素取大，无中间 tensor、无近似误差，结果与 PyTorch relu 严格一致；计算长度按 32 字节向上对齐，回写只搬有效元素，对齐补位不污染输出，错误占比 0.00%。

**内存与同步**：GM 经 GlobalTensor 绑定到本核偏移区间，UB 由 TPipe 在 Init 时一次性 InitBuffer，循环中只做 AllocTensor/EnQue/DeQue/FreeTensor。同步依赖队列语义：入队后才出队计算，结果入队后才搬出，搬完即释放，避免读写竞争；tiling 数据经 GET_TILING_DATA_WITH_STRUCT 下发，dtype 通过 schMode 型 tiling key 映射模板实例（float16 走 MODE_0、float32 走 MODE_1）。
