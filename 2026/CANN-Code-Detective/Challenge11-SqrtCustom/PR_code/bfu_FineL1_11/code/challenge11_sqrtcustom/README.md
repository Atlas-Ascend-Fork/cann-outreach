# challenge11_sqrtcustom

提交团队: 北京林业大学
提交者: FineL1


## 算子实现介绍

# 北京林业大学

## 团队信息

- 提交者: FineL1 ([L18931862188@163.com](mailto:L18931862188@163.com))
- 身份: 学生
- 单位: 北京林业大学

## 成员

- FineL1 (FineL1): 提交者, [L18931862188@163.com](mailto:L18931862188@163.com)

## 算子: SqrtCustom

实现 Ascend C Sqrt 平方根算子，支持 float16 / float32 数据类型，适配昇腾 910B4 NPU。算子针对输入形状为 (8, 2048) 的非负浮点张量，实现逐元素平方根计算 y = √x。完成输出形状推导、Tiling 数据配置以及 Kernel 侧的数据搬运与计算流程。Kernel 采用 CopyIn → Compute → CopyOut 的处理方式，通过 TQue 和 LocalTensor 管理 UB 数据，利用 Ascend C Sqrt 向量计算接口完成平方根运算，并将计算结果从 UB 搬回 GM。Host 侧根据输入元素总数、设备可用 AI Vector Core 核数及数据对齐要求进行任务切分，减少不必要的数据搬运与调度开销，优化小规模张量的计算效率，完成 Sqrt 平方根算子的开源仓规范化实现。

CANNJudge 测试已通过，验证结果为 PASS。
