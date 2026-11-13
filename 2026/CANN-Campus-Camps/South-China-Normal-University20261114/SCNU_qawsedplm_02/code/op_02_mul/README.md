# op_02_mul

提交团队: 华南师范大学
提交者: 陈俊吉


## 算子实现介绍

算子实现介绍
1.核心实现：基于 Ascend C 框架，完成了 Host 侧 Tiling 和 Kernel 侧核函数的开发。Kernel 侧采用模板化设计，通过 DT_X 自动适配 float32 与 float16 数据类型。
2.多核切分：Host 侧通过 PlatformAscendC 获取 AIV 核数，根据总长度计算每个核负责的 blockLength。为了满足硬件要求，切分步长均向上对齐到 32B（BLOCK_SIZE）。
3.UB 切分与流水：核内进一步切分为 tileNum=8 块，通过 TPipe 和 TQue 开启双缓冲（BUFFER_NUM=2）。在循环中实现 CopyIn -&gt; Compute -&gt; CopyOut 流水线，利用 AscendC::Mul 接口完成逐元素乘法，掩盖访存延迟。
4.Tiling 策略：输入总长度对齐到 32B 的整数倍计算；Tiling 结构体自动传入 Kernel 侧，为启动参数提供精准数据。
