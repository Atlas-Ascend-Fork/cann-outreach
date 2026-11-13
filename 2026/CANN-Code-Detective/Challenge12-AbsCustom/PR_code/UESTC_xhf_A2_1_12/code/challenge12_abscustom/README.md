# challenge12_abscustom

提交团队: 电子科技大学
提交者: 徐海峰


## 算子实现介绍

实现 float32/float16 逐元素绝对值：按 AIV 核数切分，依据 UB 容量双缓冲分块，用 AscendC::Abs 计算并通过 DataCopyPad 搬运尾块。CANNJudge 1206877：2/2 PASS，错误占比均为 0.00%。AI 参与：OpenAI Codex，GPT-6。
