# op_04_relu

提交团队: 华南师范大学
提交者: 崔义彬


## 算子实现介绍

**算子实现介绍**

本算子基于 Ascend C 在昇腾 NPU 上实现 ReLU 激活函数，逐元素计算 y = max (0, x)，支持 float32 /float16、ND 格式，输入输出形状一致（(8, 2048) 或 (2048, 2048)）。Host 侧 Tiling 先读取可用 AIV 核数与 UB 容量，将输入逐维累乘得到总元素数，按核数向上取整均分到各核，并按 UB 容量与 32 字节对齐折算单核单次循环处理的元素数，再依据数据类型选择 half /float 模板分支下发。Kernel 侧采用 “搬入 — 计算 — 搬出” 三级流水与双缓冲：CopyIn 将当前分块的 x 从 Global Memory 搬入 Unified Buffer，Compute 出队后调用 Relu（等价于 Maxs (x, 0)）完成逐元素取正，结果入队，CopyOut 将结果回拷 Global Memory；计算长度按 32 字节向上对齐，尾块由 DataCopyPad 处理，避免越界，输入、输出 tensor 在使用完毕后及时释放复用。
