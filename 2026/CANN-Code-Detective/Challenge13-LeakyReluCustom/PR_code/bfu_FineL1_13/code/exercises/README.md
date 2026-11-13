# exercises

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

## 算子: LeakyReLUCustom

实现基于 Ascend C 的 LeakyReLU 激活算子，支持 float16 / float32 数据类型。按照开源仓规范完成算子属性注册、输出形状推导及 Tiling 数据配置。在 op_def 中注册属性参数 `negative_slope`，由 Tiling 函数读取该属性并写入 TilingData，传递至 Kernel 侧使用，实现负半轴斜率的动态配置。Kernel 采用 CopyIn → Compute → CopyOut 的流水处理方式，通过 TQue 和 LocalTensor 管理 UB 数据，实现输入数据从 GM 到 UB 的搬运、逐元素计算及结果回写，并利用双缓冲机制提高数据搬运与计算效率。计算阶段基于 Ascend C LeakyRelu 接口实现：
y = max(x, 0) + negative_slope × min(x, 0)
其中 `negative_slope` 默认值为 0.01。算子面向昇腾 910B4 平台，支持题目指定的 (8, 2048) 输入张量，实现 LeakyReLU 激活功能的开源仓规范化开发。
