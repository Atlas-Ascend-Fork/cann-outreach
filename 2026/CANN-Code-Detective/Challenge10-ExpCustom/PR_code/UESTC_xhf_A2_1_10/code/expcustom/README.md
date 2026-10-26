# expcustom

提交团队: 电子科技大学
提交者: 徐海峰


## 算子实现介绍

实现 float32/float16 的逐元素 Exp。Host 根据 AIV 核数划分 (8, 2048) 输入，按输入 dtype 和 UB 容量确定 ubFactor，采用整数模板参数选择 float/half 实例。Kernel 使用 CopyIn → AscendC::Exp → CopyOut 流水线及双缓冲，输出 shape/dtype 与输入一致。CANNJudge 提交 921694：2/2 Pass，输出错误占比均为 0.00%。
AI 辅助：OpenAI Codex；模型 GPT-5、GPT-6；用于代码补全、编译错误分析和验证。
