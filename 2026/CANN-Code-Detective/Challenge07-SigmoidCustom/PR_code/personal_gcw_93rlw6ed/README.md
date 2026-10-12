# personal

## 团队信息

- 提交者: gcw_93rlw6ed
- 身份: 学生
- 单位: personal

## 成员

- gcw_93rlw6ed (gcw_93rlw6ed): 提交者

## 算子: challenge07_sigmoidcustom

多核数据切分：blockFactor 按 CeilDiv/CeilAlign 16 元素对齐，尾核 lastBlockLength 处理余数，小张量自动缩减核数；核内按 ubFactor 分块双缓冲流水，UB 预算 input+output+fp32 中间缓冲共 16B/元素不超限；Sigmoid 计算统一 FP32 中间精度（Muls(-1)-&gt;Exp-&gt;Adds(1)-&gt;Duplicate(1)/Div），规避 fp16 Exp 溢出与 Reciprocal 近似误差
