# op_04_relu

提交团队: 华南师范大学
提交者: 李明皓


## 算子实现介绍

算子功能
基于 Ascend C 编程语言实现的 ReLU 激活算子，在昇腾 NPU 上计算 y = max(0, x)，对应 PyTorch 原生算子 torch.nn.functional.relu。输入 x 与输出 y 均为 ND 张量，支持形状 (8, 2048) / (2048, 2048)，支持 float32 / float16。

实现方式
采用开源仓规范化算子工程，Host 侧负责 Tiling 计算，Kernel 侧模板类 NsRelu::Relu&lt;T&gt; 负责实际计算，通过 tilingKey 模板机制（RELU_TPL_SCH_MODE_0/1）按输入 dtype 自动分发到 half / float 实例。

核心设计
1. Tiling 三参数（Host 侧）

totalNum：输入 shape 逐维相乘得到总元素数（最大 2048×2048 = 4194304）；
blockFactor：每核处理元素数 = ⌈totalNum / AIV 核数⌉，按实际使用核数 ⌈totalNum / blockFactor⌉ 启动，尾核余量由 kernel 侧截断；
ubFactor：单次 UB 循环处理元素数，按 UB 容量上限计算（1 输入 + 1 输出 × 双缓冲 = 4 块 UB tensor），向下对齐到 32 字节。
2. 多核并行与尾核截断（Kernel 侧 Init）
每核按 blockIdx × blockFactor 偏移 Global Memory 指针划分数据段；尾核用 totalNum − blockIdx × blockFactor 截断实际长度，保证不越界读写。

3. 计算逻辑（Kernel 侧 Compute）
使用 Ascend C 内置矢量指令 Relu(yLocal, xLocal, calcCount) 一条指令完成 y = max(0, x)，无需中间临时缓冲。计算长度向上对齐到 32 字节（fp32 为 8 的倍数、fp16 为 16 的倍数）满足矢量指令对齐要求，多余部分位于 buffer 范围内，CopyOut 只回拷有效元素。

4. 性能优化

Double Buffer 双缓冲：输入/输出队列各 2 块 buffer（BUFFER_NUM = 2），CopyIn、Compute、CopyOut 的搬移与计算重叠执行，隐藏 GM↔UB 搬移延迟；
ubFactor 最大化：按 UB 容量上限（4 份 buffer）取单次搬移粒度（fp32 约 12288 个元素），减少每核循环次数，对 2048×2048 大输入效果显著；
DataCopyPad 按字节搬移：尾块任意长度安全搬运，无需对齐填充拷贝。
