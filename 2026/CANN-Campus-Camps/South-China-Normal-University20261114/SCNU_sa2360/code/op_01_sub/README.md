# op_01_sub

提交团队: 华南师范大学
提交者: sa2360


## 算子实现介绍

Ascend C 矢量减法采用 8 个 AI Core 并行处理，每个核按 tile 分块搬运 x、y 到 UB，使用 AscendC::Sub 计算并将结果写回 z。每个 tile 处理 128 个元素：float32 占 512B，float16 占 256B，均满足 UB 按 32B 对齐。Tiling 根据输入张量形状计算 totalLength；本题 totalLength=16384，可被 blockDim × tileNum × BUFFER_NUM × 128 整除，确保各核和各 tile 等长、无尾块。支持 float32 和 float16。
