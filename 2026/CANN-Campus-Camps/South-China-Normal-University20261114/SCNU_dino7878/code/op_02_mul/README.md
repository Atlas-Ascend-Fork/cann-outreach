# op_02_mul

提交团队: 华南师范大学
提交者: dino7878


## 算子实现介绍

基于 Ascend C 实现 z=x*y 的逐元素矢量乘法 Mul，输入/输出 shape 均为 (8,2048)，支持 float32/float16。Host 侧计算 totalLength=16384，tileNum=8，启动 8 核，workspace=0，并完成 InferShape/InferDataType。Kernel 采用模板 KernelMul&lt;DT_X&gt;，按 TilingKey 选择 float16/float；Init 计算 blockLength=2048、tileLength=128 并设置 GlobalTensor 与 UB 队列，Process 循环 16 次执行 CopyIn→Compute→CopyOut：DataCopy 搬入 x/y，AscendC::Mul 计算，DataCopy 搬出 z。固定 shape 无尾块，fp32/fp16 单块 512B/256B 均满足 32B 对齐，验证集 PASS。
