# 个人

## 团队信息

- 提交者: Murasame_214
- 身份: 学生
- 单位: 个人

## 成员

- Murasame_214 (Murasame_214): 提交者

## 算子: challenge06_relucustom

基于 Ascend C 完成逐元素 ReLU 激活算子（y = max(0, x)，支持 float32/float16）。op_host 侧 Tiling 按输入总长度与 AI Core 数多核切分、核内按 UB 容量分块；op_kernel 侧采用 CopyIn -&gt; AscendC::Relu -&gt; CopyOut 流水，多核并行。已在 CANNJudge Challenge06-ReluCustom 通过验证（PASS，precision_ratio=1.0）。
