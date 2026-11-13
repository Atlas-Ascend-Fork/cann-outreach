# op_03_tanh

提交团队: 华南师范大学
提交者: 李明皓


## 算子实现介绍

算子功能
基于 Ascend C 编程语言实现的 Tanh 激活算子，在昇腾 NPU 上计算 y = tanh(x) = (exp(x) − exp(−x)) / (exp(x) + exp(−x))，对应 PyTorch 原生算子 torch.tanh。输入 x 与输出 y 均为形状 (8, 2048) 的 ND 张量，支持 float32 / float16。

实现方式
采用开源仓规范化算子工程，工程包含 Host 侧（算子定义、形状推导、Tiling 计算）与 Kernel 侧（AI Core 核函数实现），并配套 aclnn 调用样例与单元测试。通过 tilingKey 模板机制（TANH_TPL_SCH_MODE_0/1）按输入 dtype 自动分发到 half / float 实例。

核心设计
1. Tiling 三参数（Host 侧）

totalNum：输入 shape 逐维相乘得到总元素数（16384）；
blockFactor：每核处理元素数 = ⌈totalNum / AIV 核数⌉，向上取整，按实际使用核数 ⌈totalNum / blockFactor⌉ 启动，尾核余量由 kernel 侧截断；
ubFactor：单次 UB 循环处理元素数，按 UB 容量上限计算（2 个队列 × 双缓冲 + 4 个中间结果缓冲共 8 份 UB tensor），并向下对齐到 32 字节。
2. 多核并行与尾核截断（Kernel 侧 Init）
每核按 blockIdx × blockFactor 偏移 Global Memory 指针划分数据段；尾核用 totalNum − blockIdx × blockFactor 截断实际长度，保证不越界读写。

3. Tanh 公式分解（Kernel 侧 Compute）
用 Ascend C 基础指令组合实现，4 个 TBuf&lt;VECCALC&gt; 临时缓冲常驻 UB：

Exp(tmp0, x) → eˣ
Muls(tmp1, x, −1) → −x
Exp(tmp2, tmp1) → e⁻ˣ
Add(tmp3, tmp0, tmp2) → eˣ + e⁻ˣ
Sub(tmp1, tmp0, tmp2) → eˣ − e⁻ˣ
Div(y, tmp1, tmp3) → tanh(x)
4. 对齐与尾块处理

矢量计算指令要求计数按 32 字节对齐（fp32 为 8 的倍数、fp16 为 16 的倍数），Compute 中 calcCount 向上对齐后计算，多余部分位于 buffer 范围内，CopyOut 只回拷有效元素；
CopyIn / CopyOut 使用 DataCopyPad 按字节数搬移（blockLen = currentNum × sizeof(T)），支持任意长度的尾块；
Process 按 ubLength_ 分块循环，尾块以 currentNum 指定实际元素数，避免越界。
