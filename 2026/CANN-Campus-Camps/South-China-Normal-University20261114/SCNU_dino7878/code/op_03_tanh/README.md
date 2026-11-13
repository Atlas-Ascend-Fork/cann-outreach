# op_03_tanh

提交团队: 华南师范大学
提交者: dino7878


## 算子实现介绍

基于 Ascend C 实现 y=tanh(x) 的逐元素激活算子 Tanh，输入/输出 shape 均为 (8,2048)，支持 float32/float16。Host 侧按 dtype 选择模板实例，计算 totalLength=16384，tileNum=8，启动 8 核，workspace=0。Kernel 采用模板 Tanh&lt;DT_X&gt;，BUFFER_NUM=2；Init 计算 blockLength=2048、tileLength=128 并初始化 GlobalTensor、输入输出队列和 4 个 UB 临时缓冲；Process 循环 16 次执行 CopyIn→Compute→CopyOut：DataCopy 搬入 x，Exp/Muls/Exp/Add/Sub/Div 完成 tanh 计算，DataCopy 搬出 y。固定 shape 无尾块，fp32/fp16 单块 512B/256B 均满足 32B 对齐，验证集 PASS。
