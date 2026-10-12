# 北京林业大学

## 团队信息

- 提交者: 李林阳
- 身份: 学生
- 单位: 北京林业大学

## 成员

- 李林阳 (FineL1): 提交者

## 算子: challenge07_sigmoidcustom

北京林业大学
团队信息
提交者: FineL1 (L18931862188@163.com)
身份: 学生
单位: 北京林业大学
成员
FineL1 (FineL1): 提交者, L18931862188@163.com
算子: SigmoidCustom
【代码侦探Challenge07】实现 Ascend C Sigmoid 激活算子，支持 float16 / float32 数据类型。完成输出形状推导、Tiling 数据配置以及 Kernel 侧的数据搬运与计算流程；Kernel 采用 CopyIn → Compute → CopyOut 的流水处理方式，通过 TQue 和 LocalTensor 管理 UB 数据，并使用双缓冲提高数据搬运与计算效率。计算阶段基于 Ascend C Sigmoid 接口实现 y = 1 / (1 + exp(-x))，最终将计算结果从 UB 搬回 GM，完成 Sigmoid 激活算子的开源仓规范化实现。

CANNJudge 测试已通过，验证结果为 PASS。
