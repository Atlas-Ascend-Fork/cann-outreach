# 黑龙江东方学院

## 团队信息

- 提交者: 梧知-Wisdom
- 身份: 学生
- 单位: 黑龙江东方学院

## 成员

- 梧知-Wisdom (2301_81591329): 提交者

## 算子: challenge08_tanhcustom

基于 Ascend C 实现 Tanh 激活算子。首先将输入数据从 Global Memory 搬运到 Local Memory；对于 FP16 输入，先转换为 FP32 进行中间计算，以提升数值精度。随后分别通过 Exp 计算 exp(x) 和 exp(-x)，使用 Sub 计算分子 exp(x)-exp(-x)，使用 Add 计算分母 exp(x)+exp(-x)，最后通过 Div 得到 tanh(x)。计算完成后将结果写回 Global Memory。算子结合 Tiling 和多核并行，将 (8, 2048) 输入划分给多个 AI Vector Core 处理，支持 float16 和 float32 数据类型。
