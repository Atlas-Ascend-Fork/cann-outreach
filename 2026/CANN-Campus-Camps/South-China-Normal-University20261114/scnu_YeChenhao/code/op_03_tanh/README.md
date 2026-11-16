# op_03_tanh

提交团队: 华南师范大学
提交者: gcw_Zqa1u7nr


## 算子实现介绍

本实现完成 PyTorch 原生算子 torch.tanh 的 Ascend C 版本：输入 x 与输出 y 均为 (8, 2048) 的 ND 张量，支持 float32 与 float16，共 16384 个元素。工程遵循开源仓算子规范：op_host 侧负责 Tiling 并注册算子；op_kernel 侧由 tanh.cpp 作为入口，按 tiling key 的 schMode 分流到 NsTanh::Tanh&lt;half&gt; 与 NsTanh::Tanh&lt;float&gt; 两个实例；具体计算封装在 tanh.h 的模板类里。

计算采用恒等式 tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)，整个 kernel 只有 5 遍向量指令：`Muls` 求 2x、`Exp` 求 e^(2x)、两次 Adds 分别得到分子与分母、最后 Div 完成除法。官方 ops-math 的 tanh 采用“9 阶多项式 + exp + 饱和 + 符号位拼接”的混合方案，共十余条指令，目的是与 PyTorch 逐位对齐；本题验收只需相对误差 1e-4、绝对误差 1e-4，因此选择指令数最少的形式。经 numpy 模拟验证，float32 最大绝对误差 1.2e-7，约 1 个 ULP。

半精度输入不做原生 half 计算，而是用 Cast 提升到 fp32 计算，再用 CAST_RINT 转回。原因是输入范围可达 ±10，`e^(2×10) = 4.85e8` 远超 fp16 上限 65504；原生 half 会在 |x| &gt; 5.5 处溢出为 inf，导致 inf/inf 得到 NaN；模拟中过半元素因此超差。改用 fp32 中间计算后，fp16 输出与 numpy 参考实现逐元素完全一致。

数据切分全部前移到 host 侧：按 32B 对齐（float 为 8 元素、half 为 16 元素一块）算出每核块数，只向 kernel 传总元素数与每核元素数两个整数，kernel 内不做除法、取模与循环。每核只做一次大块搬运，UB 使用两块 fp32 静态缓冲（half 额外一块半精度缓冲），通过 TBuf 直接取用，省去 TQue 的 AllocTensor、`EnQue`、`DeQue`、`FreeTensor` 事件开销；同步只保留 MTE2→V 与 V→MTE3 两对硬件事件，分别保证搬入完成后再计算、计算完成后再搬出。

性能上先用 32B 对齐整块切分消除小循环气泡，再扫描启动核数。实测 8 核慢于 4 核，说明在该规模下，每个 block 的固定开销（调度、tiling 读取、同步）超过数据搬运与计算本身，因此最终固定为 4 核。
