# op_03_tanh

提交团队: 华南师范大学
提交者: gcw_v1eirdL4


## 算子实现介绍

基于 Ascend C 在 AIV 实现 Tanh：Tiling 按 32B 对齐多核均分、核内 8 块双缓冲；fp16
先转 fp32 计算再转回，并对 x 做 ±8 截断，避免抵消误差与 exp 溢出，精度满足
rtol=1e-3、atol=1e-5。
