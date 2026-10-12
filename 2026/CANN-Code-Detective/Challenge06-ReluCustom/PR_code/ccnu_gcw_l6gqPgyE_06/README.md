# 华中师范大学

## 团队信息

- 提交者: hibert angry
- 身份: 学生
- 单位: 华中师范大学

## 成员

- hibert angry (gcw_l6gqPgyE): 提交者

## 算子: challenge06_relucustom

【代码侦探Challenge06】完成 ReLU 自定义算子

基于 CANNJudge 本期平台模板，补全 op_host/relu_tiling.cpp 和 op_kernel/relu.h，实现 float32/float16 输入的逐元素 y=max(0,x)。Tiling 根据输入类型、向量核数和 UB 容量生成按 32 字节对齐的分块参数；核函数采用 CopyIn → AscendC::Relu → CopyOut，并处理非对齐尾块。

验证：CANNJudge 提交 275777，2/2 测试点 Pass，输出错误占比均为 0.00%；平台报告用时分别为 4.86 μs 和 3.31 μs。
评测记录：https://cannjudge.cn/cann_code_detective/challenge06-relucustom/challenge06-relucustom/submission/6aa415bd2d3dd2c5aeca55b3
活动公告：https://gitcode.com/cann/cann-learning-hub/issues/730

AI 参与：是。AI Agent 平台：OpenAI Codex；AI 模型：GPT-6。根据活动任务书和平台原始模板，AI 协助补全实现、核对代码。未声称进行本地 CANN 构建或平台测试点之外的实机验收。
