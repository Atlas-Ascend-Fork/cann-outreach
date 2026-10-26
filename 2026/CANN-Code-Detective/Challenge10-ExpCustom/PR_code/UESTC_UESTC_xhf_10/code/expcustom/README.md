# expcustom

提交团队: 电子科技大学
提交者: UESTC_xhf


## 算子实现介绍

【代码侦探Challenge10】Exp 自然指数算子。按 AIV 核数切分，32B 对齐；UB 输入/输出双缓冲，DataCopyPad 处理尾块，AscendC::Exp 支持 float16/float32。CANNJudge 提交 921877：2/2 Pass，输出错误占比均为 0.00%。基于官方模板和 Add 参考实现，开发时对照公开 PR !4427、!4433；使用 OpenAI Codex 辅助实现、检查及调试。
