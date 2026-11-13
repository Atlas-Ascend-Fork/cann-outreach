# exercises

提交团队: 电子科技大学
提交者: UESTC_xhf


## 算子实现介绍

Ascend C LeakyReLU：negative_slope 属性从 host 下发到核函数；FP16/FP32 多核分块、双缓冲、非对齐尾块。CANNJudge 1257996：2/2 Pass，误差均 0.00%。AI 辅助：OpenAI Codex GPT-6。
