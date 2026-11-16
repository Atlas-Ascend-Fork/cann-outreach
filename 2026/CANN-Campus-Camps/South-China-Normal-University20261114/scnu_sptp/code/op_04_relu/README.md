# op_04_relu

提交团队: 华南师范大学
提交者: 刘柏麟


## 算子实现介绍

ReLU 算子实现 y = max(0, x)，采用 Ascend C 标准矢量编程范式，按数据量大小分流：小 shape 走官方大核/小核切分模式（4 核，按 32B 对齐将数据划分为块，前 tailBlockNum 个核多处理一块实现负载均衡，核内按 UB 容量分批循环、最后一块用尾块数量精确处理），大 shape 走 16 核整除流水模式（每核均分数据，单次搬运 8192 元素即 32KB 以充分利用带宽，配合双缓冲使搬入、计算、搬出三级流水重叠）；计算统一用 AscendC::Maxs(dst, src, 0, len) 单条向量指令完成，全程通过 TQue 队列的 EnQue/DeQue 做任务间同步，tiling 侧用 GetCoreNumAiv/GetCoreMemSize 动态获取平台核数与 UB 大小以保证泛化性。
