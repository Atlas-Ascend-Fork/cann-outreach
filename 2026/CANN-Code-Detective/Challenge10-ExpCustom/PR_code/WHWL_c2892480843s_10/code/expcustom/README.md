# expcustom

提交团队: 武汉文理学院
提交者: yunshang


## 算子实现介绍

基于 CANNJudge Challenge10 在线模板完成 Exp 自然指数算子，支持 ND 格式 (8, 2048) 的 float32/float16 输入。保留平台原有分块数据结构、调度键和核函数入口；主机侧最多使用 8 核，每次处理 64 个元素；核函数通过双缓冲队列执行搬入、AscendC::Exp 计算、搬出。CANNJudge 提交 912245：Pass，2/2 验证点通过，输出错误占比均为 0.00%。本地对实际提交源码运行受限 CPU API 模型，72 组检查通过，不等同于 CANN 官方仿真或 NPU 验证。AI 参与：使用 OpenAI Codex 协助实现与复核。
