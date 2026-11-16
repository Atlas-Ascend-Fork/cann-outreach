# op_01_sub

提交团队: 华南师范大学
提交者: 刘柏麟


## 算子实现介绍

sub_custom 是基于 Ascend C 实现的逐元素矢量减法算子，完成 z = x - y。算子采用模板化设计，支持 float 与 half 类型，Host 侧根据输入 dtype 选择对应 Kernel 实例。

Kernel 主体封装在 KernelSub&lt;T&gt; 中，采用 CopyIn → Compute → CopyOut 三段式流水：通过 DataCopy 将 Global Memory 数据搬入 Local Memory，调用 AscendC::Sub 完成矢量减法，再将结果搬回 Global Memory。输入输出使用 TQue 队列管理，BUFFER_NUM = 2 实现 Double Buffer，使搬运与计算形成 Ping/Pong 流水。

多核切分采用“基础长度 + 余数分摊”策略：Host 侧计算 blockLength = totalLength / blockDim 和 tailLength = totalLength % blockDim，前 tailLength 个核各多处理 1 个元素，保证数据完整覆盖并尽量均衡负载。每个核根据 blockIdx 计算自身偏移和 currentLength，并据此动态计算 tileLength，向下对齐到 32 字节，满足 DataCopy 对齐要求。

主循环次数为 currentLength / tileLength，不再将 BUFFER_NUM 乘入循环次数，Double Buffer 由队列深度自动生效。主循环处理对齐块，剩余非对齐尾块通过标量 GetValue/SetValue 兜底，避免非对齐 DataCopy 的未定义行为。TPipe 在核函数入口创建并传入 KernelSub，减少重复构造开销。

整体实现兼顾正确性、多核负载均衡、搬运对齐与流水效率，结构清晰，便于扩展数据类型和后续调优。
