# op_01_sub

提交团队: 华南师范大学
提交者: frost35_snow


## 算子实现介绍

基于 Ascend C 实现 z = x - y，(8, 2048)、ND、fp32/fp16。8 核 &lt;&lt;&lt;&gt;&gt;&gt; 启动，GlobalTensor 按 blockIdx 切段，TPipe/TQue 走 CopyIn → Compute → CopyOut 三级流水，计算用 AscendC::Sub。
把16次细分块改为按 UB 预算一次搬完每核 2048 元素：QUEUE_DEPTH = 1 时 Double Buffer 无法重叠，原实现多付 15 轮搬运与同步，改后全局 DMA 384 → 24 次、Sub 发射 128 → 8 次。Tiling 按 96KB 预算与 32 字节对齐自适应，兼容不同 shape。
