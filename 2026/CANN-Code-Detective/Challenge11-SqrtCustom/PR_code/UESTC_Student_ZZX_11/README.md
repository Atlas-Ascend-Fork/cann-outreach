# 电子科技大学

## 团队信息

- 提交者: Student_ZZX
- 身份: 学生
- 单位: 电子科技大学

## 成员

- Student_ZZX (Student_ZZX): 提交者

## Challenge 11: SqrtCustom

- 提交者：Student_ZZX（电子科技大学，UESTC）
- 算子：Ascend C Sqrt，支持 float16 / float32，输入为非负数
- 实现：Host 侧按 shape、AI Vector 核数和 UB 容量切分；Kernel 侧使用双缓冲、DataCopyPad 处理尾块并执行 `AscendC::Sqrt`
- 验证：CANNJudge 提交 992957，4/4 测试点 Pass；本地已检查补丁格式与 Python 数据生成脚本语法。PR 更新后的开源仓完整构建以自动检查为准。

AI 参与：是。AI Agent 平台：OpenAI Codex；模型：GPT-6。使用 AI 协助补全官方模板、修复评审指出的示例与单测问题、核对提交。提示词概要：基于第十一期官方任务和 CANNJudge 模板实现 Sqrt 算子，通过评测并按开源仓规范提交。
