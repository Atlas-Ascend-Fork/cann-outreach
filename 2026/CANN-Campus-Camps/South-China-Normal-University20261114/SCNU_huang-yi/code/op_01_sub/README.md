# op_01_sub

提交团队: 华南师范大学
提交者: gcw_nTAbyG7d


## 算子实现介绍

矢量减法算子（sub）实现要点
功能：z = x - y，逐元素减法；输入 x,y 为 (8,2048)，支持 float32/float16。

代码结构（kernel.asc 直调形态）：

run_kernel（host 侧）：算 totalLength，按 dtype 选 sub_custom&lt;float&gt; / sub_custom&lt;half&gt; 启动，硬编码 blockDim=8。
sub_custom&lt;T&gt;（device 侧核函数）：调 KernelSub&lt;T&gt; 的 Init + Process。
关键设计

多核切分：blockLength = totalLength / GetBlockNum() = 16384/8 = 2048；每核用 GetBlockIdx() 把 GM 地址偏移 2048×k，只搬/写自己那段，8 核并行不重叠。
单核再切 tile：tileNum=8、BUFFER_NUM=2 → tileLength = 2048/8/2 = 128，loopCount = 8×2 = 16（16×128=2048，正好覆盖）。
双缓冲流水：每个 tile 走 CopyIn(GM→UB) → Compute(Sub) → CopyOut(UB→GM)，两块 buffer 让搬运与计算重叠，UB 不空转。
泛型类型：模板 KernelSub&lt;T&gt;，T=float/half，计算时调 AscendC::Sub。
