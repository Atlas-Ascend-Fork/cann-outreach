# expcustom

提交团队: 电子科技大学
提交者: StudentXHF


## 算子实现介绍

【代码侦探Challenge10】Exp 自然指数算子。支持 float32/float16，8核均分，每核2048元素，UB每次处理256元素，使用 AscendC::Exp 和队列完成 GM→UB→GM。CANNJudge 提交921735验证通过，2/2测试点PASS。AI参与：是；AI Agent平台：OpenAI Codex；AI模型：GPT-6；Prompt上下文：基于官方模板补全Exp算子的host tiling和kernel，核对平台模板接口，并通过CANNJudge验证。

本次活动的验证输入限定为 ND 形状 (8, 2048)，数据类型为 float32 或 float16。实现按完整的 256 元素 tile 处理，不支持非 256 整数倍的元素总数，也不声明支持任意形状或尾块。
