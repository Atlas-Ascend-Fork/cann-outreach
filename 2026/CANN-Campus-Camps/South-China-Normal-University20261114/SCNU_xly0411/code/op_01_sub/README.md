# op_01_sub

提交团队: 华南师范大学
提交者: 徐琳雅


## 算子实现介绍

本算子基于 Ascend C 实现矢量减法 z = x − y。将 16384 个元素（8×2048）按"8 核并行 + 单核 8 块 + 双缓冲"切分：每核负责 2048 个元素，再切成 128 元素的小块循环处理。每块经 CopyIn（DataCopy 搬入 UB）→ Compute（`AscendC::Sub` 矢量指令完成逐元素相减）→ CopyOut（DataCopy 写回 GM）三级流水，配合 BUFFER_NUM=2 双缓冲使搬运与计算重叠执行。核函数通过模板 `&lt;typename T&gt;` 泛化，host 侧按 dtype 实例化 `sub_custom&lt;half&gt;` 或 `sub_custom&lt;float&gt;`，一份代码适配 float32/float16。
