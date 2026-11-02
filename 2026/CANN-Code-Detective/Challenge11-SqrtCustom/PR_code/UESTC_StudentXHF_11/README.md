# Challenge11 SqrtCustom

- 学校：电子科技大学（UESTC）
- 提交者：StudentXHF
- 个人标识：StudentXHF_11
- 工程：`code/sqrtcustom`

按活动指定的 ND `(8, 2048)`、float32/float16 输入实现逐元素 `Sqrt(x)`，输入非负。Host tiling 在 Ascend 910B 上将 16384 个元素分给 8 个核，每核 2048 个元素，每次搬运 256 个元素；kernel 通过双缓冲队列执行 `CopyIn → Sqrt → CopyOut`。其它形状仅在元素数及分核结果满足 256 元素对齐时适用，未声明通用尾块支持。

[CANNJudge 提交 992965](https://cannjudge.cn/cann_code_detective/challenge11-sqrtcustom/challenge11-sqrtcustom/submission/6ae1e0ed1dac5d235b5ec4d9)：4/4 测试点 PASS，输出错误占比均为 0.00%。这是平台针对题目验证集的结果；仓库模板中的独立 host UT、NPU 示例没有另行运行。

AI 参与：是。AI Agent 平台：OpenAI Codex；模型：GPT-6。Prompt 上下文：依据 Challenge11 官方 Sqrt 模板的 Tiling 和 kernel TODO，完成多核切分、UB 队列搬运与平方根计算，并用 CANNJudge 验证后提交。
