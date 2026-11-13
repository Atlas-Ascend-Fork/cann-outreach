# op_04_relu

提交团队: 华南师范大学
提交者: ljt


## 算子实现介绍

本算子基于 Ascend C 编程语言原生开发，在昇腾 NPU 硬件上实现 ReLU 激活算子功能（y = max(0, x)）。具体实现策略如下：
Tiling 策略与 32B 对齐：在 Host 侧 Tiling 中，根据硬件平台获取可用 AI Core 数量与 UB 大小。通过 GetUbBlockSize 动态计算 32 字节对齐的元素个数（alignElements），总数据量按核数切分并向上对齐至 32B 的整数倍（blockFactor）。为避免因对齐产生的空核，通过反算求出实际需要的核数（usedCoreNum）并配置启动。同时根据输入 dtype（float32 / float16）动态下发不同的 TilingKey。
多核切分与尾块处理：Kernel 侧通过 GetBlockIdx() 计算当前核的数据偏移。针对无法被核数整除的情况，在最后一个核通过计算 remainder 动态修正 blockLength_，保证尾部数据不丢失，且不产生越界访问，兼容 (8, 2048) 和 (2048, 2048) 等多种 Shape。
DoubleBuffer 流水线：在核内采用双缓冲（BUFFER_NUM = 2）机制，通过 TPipe 和 TQue 分配队列内存。流水化执行 CopyIn -&gt; Compute -&gt; CopyOut，利用流水线重叠掩盖搬运延迟，提升吞吐性能。
核心计算与命名空间冲突规避：在 Compute 阶段，直接调用 Ascend C 提供的高阶 API ::AscendC::Relu 完成计算。通过全局作用域符 :: 精准指定调用底层硬件 API，成功规避了自定义核函数类名 Relu 带来的 C++ 命名冲突，原生支持 float32 与 float16 两种数据类型。
