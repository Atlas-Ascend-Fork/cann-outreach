# 青海大学

## 团队信息

- 提交者: wsnidie1
- 身份: 学生
- 单位: 青海大学

## 成员

- wsnidie1 (wsnidie1): 提交者

## 算子: challenge08_tanhcustom

使用 AscendC 实现 Tanh 激活算子：host 侧 Tiling 按 8 核并行切分（向上取整、尾核截断防御、越界核空转），UB 单块搬运含容量校验；kernel 侧按 schMode 分派 fp32/fp16 双实例，采用数值稳定的 1 - 2/(e^{2x}+1) 公式（仅 1 次 Exp 运算），统一 DataCopyPad 搬运；含完整 UT 与精度比对（MERE/MARE）体系。
