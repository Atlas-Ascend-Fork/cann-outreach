# challenge11_sqrtcustom

提交团队: 电子科技大学
提交者: 徐海峰


## 算子实现介绍

实现逐元素 Sqrt：支持 float32/float16，按 AIV 核数以 256 元素对齐切分，按 UB 容量配置双缓冲，通过 DataCopyPad → AscendC::Sqrt → DataCopyPad 完成计算和尾块搬运。CANNJudge 提交 994389 已通过 4/4，输出错误占比均 0.00%。
AI 参与：OpenAI Codex；模型 GPT-6；协助补全 Host Tiling、核函数、验证及提交材料。
