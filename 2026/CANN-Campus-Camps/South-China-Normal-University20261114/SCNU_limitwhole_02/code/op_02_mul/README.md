# op_02_mul

提交团队: 华南师范大学
提交者: 李明皓


## 算子实现介绍

算子功能
基于 Ascend C 编程语言实现的矢量乘法算子，在昇腾 NPU 上计算 z = x × y，输入 x、y 与输出 z 均为形状 (8, 2048) 的 ND 张量，支持 float32 / float16。

实现方式
采用自定义算子工程模式，工程分为 Host 侧与 Kernel 侧两部分：Host 侧负责 Tiling 计算、算子注册与框架对接（InferShape/InferDataType）；Kernel 侧负责在 AI Core 上执行实际计算，数据类型通过 Tiling Key 模板机制（ASCENDC_TPL_SEL_PARAM）自动分发到 float / half 实例。

核心设计
1. Tiling 切分策略（Host 侧）
TilingFunc 从输入 shape 计算总元素数 16384，设置 tileNum = 8，通过 SetBlockDim(8) 启动 8 个 AI Core；tiling 参数（totalLength、tileNum）经 MulTilingData 结构体传递给 Kernel 侧。

2. 多核数据并行（Kernel 侧）
每核处理 totalLength / 8 = 2048 个元素，通过 GetBlockIdx() 对 Global Memory 指针做偏移划分数据段，各核互不重叠；单核内配合 BUFFER_NUM = 2 双缓冲，单次搬移粒度 128 个元素，共循环 16 次。

3. 三段式流水处理
沿用 Ascend C 标准结构：

CopyIn：DataCopy 将 x、y 数据块从 Global Memory 搬入 UB，入队；
Compute：从输入队列取出 LocalTensor，调用 AscendC::Mul(z, x, y, tileLength) 完成矢量乘法，结果入输出队列；
CopyOut：出队后将计算结果搬回 Global Memory。
4. UB 内存管理
基于 TPipe + TQue 队列管理片上统一缓冲区，三个队列（两个 VECIN 输入、一个 VECOUT 输出）各分配 2 块 buffer，AllocTensor / FreeTensor 严格配对，避免 UB 泄漏。

框架集成
InferShape：输出 shape 与输入一致（逐元素算子）；
InferDataType：输出 dtype 跟随输入；
算子注册：通过 OpDef 声明输入输出属性（REQUIRED、DT_FLOAT/DT_FLOAT16、FORMAT_ND），配置 ascend910b 平台与 Tiling 函数。
