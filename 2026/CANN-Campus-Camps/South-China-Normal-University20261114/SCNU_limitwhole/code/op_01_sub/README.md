# op_01_sub

提交团队: 华南师范大学
提交者: 李明皓


## 算子实现介绍

算子功能
基于 Ascend C 编程语言实现的矢量减法算子，在昇腾 NPU 上计算 z = x − y，输入 x、y 与输出 z 均为形状 (8, 2048) 的 ND 张量，支持 float32 / float16。

实现方式
采用直接调用（Direct Invocation）模式：host 侧 run_kernel 负责 tiling 计算与核函数发射，device 侧 sub_custom 核函数在 AI Core 上执行分块计算，无需经过算子注册与编译流水线。

核心设计
1. 多核数据并行
将 16384 个元素按 blockDim = 8 切分到 8 个 AI Core，每核通过 GetBlockIdx() 对 Global Memory 指针做偏移，处理连续的一段 2048 个元素，各核互不重叠。

2. Tiling 切分策略
每个核内进一步分 tileNum = 8 块，配合 BUFFER_NUM = 2 双缓冲，单次搬移粒度 tileLength = 128 个元素，共循环 16 次完成单核全部数据。Tiling 参数（totalLength、tileNum）由 host 侧计算后按值传入核函数。

3. 三段式流水处理
沿用 Ascend C 标准结构：

CopyIn：DataCopy 将 x、y 数据块从 Global Memory 搬入 UB，入队；
Compute：从输入队列取出 LocalTensor，调用 AscendC::Sub(z, x, y, tileLength) 完成矢量减法，结果入输出队列；
CopyOut：出队后将计算结果搬回 Global Memory。
4. UB 内存管理
基于 TPipe + TQue 队列管理片上统一缓冲区：三个队列（两个 VECIN 输入、一个 VECOUT 输出）各分配 2 块 buffer，AllocTensor / FreeTensor 严格配对，避免 UB 泄漏。
