# op_02_mul

提交团队: 华南师范大学
提交者: zhangdu


## 算子实现介绍

算子功能：实现逐元素乘法 z = x * y，支持 float32/float16，shape (8, 2048)，ND 格式。
Host 侧：根据输入 shape 计算总元素数，配置 8 个 AI Core 并行，设置 Tiling 参数（totalLength、tileNum）传入 Kernel。
Kernel 侧：采用 CopyIn → Compute → CopyOut 三段式流水线，BUFFER_NUM=2 双缓冲掩盖搬运延迟，核心计算调用 AscendC::Mul 向量指令完成逐元素乘法。
精度：输出 dtype 与输入保持一致，由 InferDataType 自动推导。
