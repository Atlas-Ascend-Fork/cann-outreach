# op_04_relu

提交团队: 华南师范大学
提交者: sa2360


## 算子实现介绍

Ascend C ReLU 采用多核并行，每核从 Global Memory 搬运分块数据到 UB，调用 Maxs(y, x, 0) 实现 y=max(0,x)，再写回输出。Tiling 根据输入形状计算 totalNum，按 32B 对齐得到每核 blockFactor，并根据 UB 容量设置对齐的 ubFactor；启动核数按 totalNum/blockFactor 计算，尾块按实际长度处理。支持 float32、float16，适配 (8,2048) 与 (2048,2048) 两种形状。
