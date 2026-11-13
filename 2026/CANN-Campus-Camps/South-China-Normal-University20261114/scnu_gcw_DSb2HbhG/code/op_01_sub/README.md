# op_01_sub

提交团队: 华南师范大学
提交者: gcw_DSb2HbhG


## 算子实现介绍

8 核并行，单核 2048 元素；核内 8×2 分块，单 Tile 128 元素（fp32 512B/fp16 256B，32B 对齐）；每 Tile 经 DataCopy 搬入 UB，AscendC::Sub 计算 z=x-y 后搬回 GM。Tiling 按 shape 动态求 totalLength，blockLength=totalLength/8，tileLength=blockLength/8/2；dtype=1 用 half，否则用 float。
