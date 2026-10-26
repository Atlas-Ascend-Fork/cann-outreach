# expcustom

提交团队: 北京林业大学
提交者: FineL1


## 算子实现介绍

北京林业大学
团队信息
提交者: FineL1 (L18931862188@163.com)
身份: 学生
单位: 北京林业大学
成员
FineL1 (FineL1): 提交者, L18931862188@163.com
算子: ExpCustom
实现 Ascend C Exp 自然指数算子，支持 float16 / float32 数据类型。完成输出形状推导、Tiling 数据配置以及 Kernel 侧的数据搬运与计算流程；Kernel 采用 CopyIn → Compute → CopyOut 的流水处理方式，通过 TQue 和 LocalTensor 管理 UB 数据，并使用双缓冲提高数据搬运与计算效率。计算阶段基于 Ascend C Exp 接口实现 \(y=e^x\)，支持输入形状为 (8, 2048) 的张量计算，最终将计算结果从 UB 搬回 GM，完成 Exp 自然指数算子的开源仓规范化实现。

CANNJudge 测试已通过，验证结果为 PASS。
