# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: 王晨宇


## 算子实现介绍

使用 TPipe + TQue 双缓冲，按 CopyIn -&gt; Sub -&gt; CopyOut 流水执行。
支持 float32 与 float16，按 dtype 分派到 sub_custom&lt;float&gt; / sub_custom&lt;half&gt;。
自动展平 (8, 2048) 的 ND 张量。
按可用 Vector Core 数量切分数据，并保证每个核起始地址 32B 对齐。
使用 DataCopyPad 处理非 32B 对齐的尾块。
单次 UB 处理块大小为 4096 个元素，适用于当前 16384 个元素的数据规模。
