# op_01_sub

提交团队: 华南师范大学
提交者: dino7878


## 算子实现介绍

基于 Ascend C 实现 z=x-y 的逐元素矢量减法 SubCustom，输入/输出 shape 均为 (8,2048)，支持 float32/float16。Host 侧统计 totalLength=16384，启动 8 核，tileNum=8；每核 blockLength=2048，按 BUFFER_NUM=2 切分，tileLength=128，循环 16 次。Kernel 采用模板 KernelSub&lt;T&gt;，Init 设置 GlobalTensor 与 UB 队列，Process 执行 CopyIn→Compute→CopyOut：DataCopy 搬入 x/y，AscendC::Sub 计算，DataCopy 搬出 z。固定 shape 无尾块，fp32/fp16 单块 512B/256B 均满足 32B 对齐，验证集 PASS。
