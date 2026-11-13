# op_03_tanh

提交团队: 华南师范大学
提交者: frost35_snow


## 算子实现介绍

Tanh 算子实现（fp32/fp16、ND、(8,2048)），补齐 Tiling 与 Kernel 全部 TODO。

改动：把 tanh 的双指数公式改写为单次 Exp 的等价式，并给 2x 加 ±40 饱和区截断，少算一次 Exp、消除大输入下的溢出；为 half 输入增加 fp32 中间计算与回写转换，避免低精度丢位；重做 Tiling，按 dtype 对齐粒度多核切分、按 UB 容量二次分块，将 4 个 fp32 临时 buffer 计入 UB 预算，两种 dtype 共用一套 kernel 模板。验证结果：fp32 最大绝对误差 1.19e-7，fp16 与参考逐位一致，±1e30 输入输出 ±1 且无 NaN，Tiling 覆盖完整无重叠、搬运全 32B 对齐。
