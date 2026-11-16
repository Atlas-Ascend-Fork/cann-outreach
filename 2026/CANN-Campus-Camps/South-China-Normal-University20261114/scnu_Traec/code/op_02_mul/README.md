# op_02_mul

提交团队: 华南师范大学
提交者: 崔义彬


## 算子实现介绍

乘法算子（Mul）实现介绍
一、算子功能
本算子基于 Ascend C 在昇腾 NPU 上实现逐元素矢量乘法，计算表达式为 z = x × y。输入张量 x、y 与输出张量 z 形状均为 (8, 2048)，支持 float32 / float16 两种数据类型，数据格式为 ND，输出 shape 与数据类型与输入 x 保持一致。
二、工程结构
算子采用 Host 侧与 Kernel 侧分离的标准 Ascend C 工程结构：
op_host（Host 侧）：负责 Tiling 切分参数计算、形状推导（InferShape）、数据类型推导（InferDataType）以及算子 OpDef 注册。
op_kernel（Kernel 侧）：KernelMul 核函数类，完成数据搬入、矢量计算、结果搬出的片内执行流程；mul_tiling.h 定义 Tiling 结构体，tiling_key_mul.h 按 dtype 选择 half / float 模板实例。
三、Host 侧 Tiling 流程
从输入张量读取数据类型，通过 ASCENDC_TPL_SEL_PARAM 完成模板参数选择；
遍历输入 shape 维度累乘得到总元素个数 totalLength，按 TILE_NUM = 8 写入 tileNum；
SetBlockDim(8) 配置启动 8 个 AIV 核并行处理，workspace 大小置 0；
InferShape 将输出 shape 置为与输入 x 相同，InferDataType 将输出 dtype 置为与输入 x 相同。
四、Kernel 侧执行流程
核函数按 CopyIn → Compute → CopyOut 三级流水循环执行：
Init：按核数计算每核处理长度 blockLength = totalLength / GetBlockNum()，再由 tileNum、BUFFER_NUM 折算出单块长度 tileLength；为 x/y/z 三个 GlobalTensor 设置各核的 GM 起始地址；通过 TPipe 为输入、输出队列分配 UB 缓冲。
CopyIn：从输入队列分配 LocalTensor，用 DataCopy 将 x、y 当前块从 Global Memory 搬入 Unified Buffer，再 EnQue 入队。
Compute：从输入队列 DeQue 取出 x、y，为结果分配 zLocal，调用矢量乘法指令 AscendC::Mul(zLocal, xLocal, yLocal, tileLength) 完成逐元素相乘；结果 EnQue 送入输出队列，并释放输入张量内存。
CopyOut：从输出队列取出结果，用 DataCopy 整块搬回 Global Memory 对应位置，释放输出张量。
五、关键设计
多核并行：8 个核按 blockLength 均分 16384 个元素，各核独立处理自己的数据区间。
队列与缓冲：inQueueX/inQueueY 位于 VECIN 位置，outQueueZ 位于 VECOUT 位置，BUFFER_NUM = 2 支撑输入/输出缓冲轮转。
类型模板化：核函数以 DT_X 为模板参数，fp16 实例化为 half、fp32 实例化为 float，由模板选择宏自动匹配。
