# op_04_relu

提交团队: 华南师范大学
提交者: dino7878


## 算子实现介绍

基于 Ascend C 实现 y=max(0,x) 的逐元素 ReLU 算子，支持 (8,2048)/(2048,2048)、float32/float16。Host 侧读取 dtype 和 UB 大小，按 4 份 UB 缓冲计算 ubFactor，动态选择 2 的幂次 tileNum，保证 tileLength 不超 UB，启动 8 核并设置 workspace=0。Kernel 采用模板 Relu&lt;DT_X&gt;，BUFFER_NUM=2；Init 计算 blockLength/tileLength、设置 GlobalTensor 偏移并初始化 UB 队列，Process 循环 tileNum*BUFFER_NUM 次执行 CopyIn→Compute→CopyOut：DataCopy 搬入 x，AscendC::Relu 计算，DataCopy 搬出 y。大 shape 下动态增大 tileNum 避免 UB 溢出，tileLength 32B 对齐且无尾块，验证集 PASS。
