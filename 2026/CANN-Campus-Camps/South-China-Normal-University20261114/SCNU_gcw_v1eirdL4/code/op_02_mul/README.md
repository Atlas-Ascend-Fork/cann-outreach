# op_02_mul

提交团队: 华南师范大学
提交者: gcw_v1eirdL4


## 算子实现介绍

这是基于 Ascend C 编程语言，在昇腾 910B 的 AI Vector Core 上原生实现逐元素矢量乘法
z = x * y，支持 float32 与 float16 两种数据类型，输入输出均为 ND 格式、形状 (8,2048)，
计算全部在片上完成，无需额外 workspace。

工程分为 host 与 kernel 两侧。Host 侧 TilingFunc 通过 PlatformAscendC 获取硬件 AIV 核数，
读入 x 的形状与数据类型，并以数据类型注册 tiling key，使同一份 kernel 模板按 DT_X 分别
实例化为 fp32 与 fp16 两个版本，元素总数写入 MulTilingData::length。由于 16384 无法被
48 整除，若直接按全部 AIV 核均分，既会遗留 16 个元素的尾部空洞，又会使每块元素数折算的
字节数不满足 DataCopy 的 32 字节对齐要求。因此 Tiling 中实现了自适应核数选择：从硬件核数
向下搜索最大的核数 c，使其同时满足总长被 c 整除、每核长度可被 TILE_NUM×BUFFER_NUM 均分、
每块字节数不小于 32 且为 32 的整数倍，命中即 SetBlockDim(c)，从而保证计算区间严格覆盖全部
16384 个元素且无尾块。InferShape 与 InferDataType 采用透传策略，输出形状与数据类型直接
继承输入 x。

Kernel 侧以模板类 KernelMul&lt;DT_X&gt; 组织。Init 用 GetBlockNum / GetBlockIdx 将全局张量按核
切分，依据 blockOffset 为 x、y、z 建立 GlobalTensor 视图，并用 TPipe 为两个输入队列和一个
输出队列各申请 BUFFER_NUM=2 块 UB 缓冲，构成双缓冲。Process 将本核数据均分为 TILE_NUM×
BUFFER_NUM=16 块，逐块执行 CopyIn→Compute→CopyOut 流水：DataCopy 把分块从 GM 搬入 UB，
DeQue 后调用 AscendC::Mul 完成向量乘，再写回 GM。TQue 的 AllocTensor / EnQue / DeQue /
FreeTensor 成对使用，缓冲在流水间复用，实现搬运与计算重叠。

本实现在多档硬件核数配置下均保持"核数 × 每核长度 = 16384"的精确覆盖，并满足矢量搬运的
对齐要求，可同时通过 float32 与 float16 验证集。
