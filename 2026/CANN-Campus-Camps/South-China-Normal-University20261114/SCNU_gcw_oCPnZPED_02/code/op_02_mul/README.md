# op_02_mul

提交团队: 华南师范大学
提交者: gcw_oCPnZPED


## 算子实现介绍

实现方案：多核并行 + 大块搬运。

Tiling 策略：host 侧从第一个输入张量取元素个数 totalLength，以每核至少处理 2048 个元素
计算实际启动核数（16384 个元素即启动 8 个核，恰好每核一行 2048 个元素），并保证每核起
始偏移按 32B 对齐；workspace 设为 0。数据类型（float32/float16）通过 tiling key 模板
参数在编译期分发，kernel 侧无需分支判断。

Kernel 实现：每个核按 GetBlockIdx 定位自己负责的连续数据段，TILE_LEN 取 4096 个元素
（float32 为 16KB，满足 32B 对齐且不超过 UB 容量），先由 MTE2 将 x、y 由 GM 搬入 UB，
再用一条 AscendC::Mul 向量指令完成整块逐元素乘法，最后由 MTE3 将结果搬回 GM。由于每核
数据量只有一个分块，采用单缓冲、一轮完成，避免多余的流水同步开销；空核直接返回。
