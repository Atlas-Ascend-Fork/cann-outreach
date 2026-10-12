# 浙江润农

## 团队信息

- 提交者: xyczc
- 身份: 企业员工
- 单位: 浙江润农

## 成员

- xyczc (xyczc): 提交者

## 算子: challenge06_relucustom

实现 float32/float16 ReLU：输出形状与输入一致，按 32 字节对齐进行多核分块，依据 UB 容量设置双缓冲和比较掩码。使用 Relu 指令计算，并通过 Compare/Select 保留 NaN，匹配 PyTorch ReLU 语义；末尾不足对齐块时使用 DataCopyPad。CANNJudge 提交 249532 的全部 2 个验证用例通过，正确比例均为 100%。
