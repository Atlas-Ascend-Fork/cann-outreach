# op_02_mul

提交团队: 华南师范大学
提交者: gggga5


## 算子实现介绍

1、Host 侧 Tiling：解析输入、输出 tensor 的维度信息 (8, 2048)，得到元素总数 16384；结合 UB 可用容量与数据类型，计算单个 tile 的长度，并按照 AI Core 数量对元素总量进行划分，进而生成 blockDim、单核处理长度、tile 个数等 tiling 参数。

2、Kernel 初始化：取得 x、y、z 对应的 GlobalTensor，并依据 tiling 参数确定当前核所负责数据的起始偏移与处理长度；通过 TPipe/TQue 在 UB 中申请缓存空间，用于存放输入与输出 tile。

3、数据搬运：对当前核负责的各个 tile 进行循环处理，利用 DataCopy 将 x、y 的相应 tile 从 Global Memory 拷贝至 UB 中的 LocalTensor；对于尾部 tile，按剩余实际长度处理，避免发生越界访问。

4、矢量乘法计算：调用 Ascend C 提供的矢量 Mul 接口，在 UB 中对 xLocal、yLocal 执行逐元素相乘，即 z = x * y，并将计算结果写入 zLocal。

5、结果搬出：通过 DataCopy 将 zLocal 写回输出 GlobalTensor z 的对应区域。

6、多数据类型支持：利用模板参数或编译宏对 float32 与 float16 进行区分，复用同一套数据搬运及 Mul 计算流程；同时保证 tile 长度、字节数及对齐方式均按对应数据类型正确计算。
