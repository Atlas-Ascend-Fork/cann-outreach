# expcustom

提交团队: 华为技术有限公司
提交者: Abeltianxiong


## 算子实现介绍

实现 PyTorch 原生算子 torch.exp （y = eˣ），基于 Ascend C 的 Vector 流水编程范式，支持 fp16/bf16/fp32 三种数据类型。
