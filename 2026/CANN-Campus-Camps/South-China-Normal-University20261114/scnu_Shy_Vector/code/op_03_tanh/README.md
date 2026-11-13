# op_03_tanh

提交团队: 华南师范大学
提交者: Shy_Vector


## 算子实现介绍

分为 host 侧与 kernel 侧两部分。
核心实现要点
1. 数学公式（tanh.h:96-107）

tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
采用一步标量 Muls + 限幅 + Exp 展开：dst=e^(2x)，分子分母分别 Adds ±1 后 Div。为防止 exp(2x) 上溢成 inf 导致 inf/inf，先对参数限幅到 [-17.6, 17.6]（tanh.h:100-102）。

2. 流水调度（tanh.h:145-159）

经典 CopyIn → Compute → CopyOut 三级流水，双缓冲（BUFFER_NUM=2），隐藏搬运与计算延迟。
输入队列 VECIN、输出队列 VECOUT、临时 VECCALC 三块内存分工。
3. 多核切分 + 循环切块（tanh_tiling.cpp:94-111）

数据量 &gt; 1024 时才多核切分（避免核间调度开销），每核 blockFactor 按 32B 对齐。
核内按 ubFactor 循环搬运计算，UB 预算取 9/10 留余量。
4. 对齐与尾块处理（tanh.h:81-93, 130-142）

常规路径走 32B 对齐的 DataCopy；尾部不足 32B 的元素走 DataCopyPad 非对齐通路，精确到字节，避免 UB 越界（预留 UB_PAD_SIZE=32）。
5. 低精度升精度计算（tanh.h:115-121）

half 场景先 Cast&lt;float&gt; 到 float 计算中间结果，最后 CAST_RINT 舍入回 half，避免精度损失；float 场景复用输入缓冲区空间存放分母。
6. 模板分派（tanh.cpp:19-28, tanh_tiling_key.h）

schMode=0 → half，schMode=1 → float，由 host 侧按 dtype 选择 tilingKey，编译期 if constexpr 分派，避免运行时分支。
