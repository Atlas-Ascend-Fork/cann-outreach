# challenge11_sqrtcustom

提交团队: 电子科技大学
提交者: UESTC_xhf


## 算子实现介绍

基于官方 Sqrt 模板与本人第十期实现补全。支持 float16/float32，AIV 多核切分、32B 对齐、输入输出双缓冲、DataCopyPad 尾块搬运，AscendC::Sqrt 逐元素求平方根；空张量直接返回，校验负维与元素乘积溢出。CANNJudge 992919：4/4 Pass，错误占比均为0.00%。AI-assisted: OpenAI Codex GPT-6；所有实现、调试和提交均有 AI 参与。
