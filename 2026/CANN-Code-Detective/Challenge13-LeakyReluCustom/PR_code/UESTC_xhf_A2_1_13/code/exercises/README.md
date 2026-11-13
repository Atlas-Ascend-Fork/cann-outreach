# exercises

提交团队: 电子科技大学
提交者: 徐海峰


## 算子实现介绍

支持 float32/float16 LeakyReLU，读取 negative_slope 后随 Tiling 数据下发，按 AIV 和 UB 分块并用双缓冲执行。CANNJudge 1257822：2/2 PASS。AI 参与：OpenAI Codex，GPT-6。
