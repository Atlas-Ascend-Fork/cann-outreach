# op_01_sub

提交团队: 华南师范大学
提交者: gcw_ov7fsX05


## 算子实现介绍

多核切分 + 核内 8×2 分块，tileLength=128 元素；TPipe/TQue 实现 CopyIn-Compute-CopyOut 三级流水，Sub 逐元素相减，按 32B 对齐分配 UB。
