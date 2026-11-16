# op_02_mul

提交团队: 华南师范大学
提交者: 钟蔚


## 算子实现介绍

1、Host 侧 Tiling：读取输入输出 tensor 形状 (8, 2048)，总元素数为 16384；根据 UB 容量和数据类型确定每个 tile 长度，并将总元素按 AI Core 数量切分，生成 blockDim、每核处理长度、tile 数量等 tiling 参数。

2、Kernel 初始化：获取 x、y、z 的 GlobalTensor，根据 tiling 参数计算当前核负责的数据偏移和长度；通过 TPipe/TQue 申请 UB 空间，用于缓存输入和输出 tile。

3、数据搬运：循环处理当前核负责的 tile，使用 DataCopy 将 x、y 对应 tile 从 Global Memory 搬运到 UB LocalTensor；尾块按实际剩余长度处理，保证不越界。

4、矢量乘法计算：调用 Ascend C 的矢量 Mul 接口，对 UB 中的 xLocal、yLocal 逐元素执行 z = x * y，结果写入 zLocal。

5、结果搬出：使用 DataCopy 将 zLocal 搬回输出 GlobalTensor z 的对应位置。

6、多数据类型支持：通过模板参数或编译宏区分 float32 和 float16，复用同一套搬运与 Mul 计算流程；确保 tile 长度、字节数和对齐按数据类型正确计算。
