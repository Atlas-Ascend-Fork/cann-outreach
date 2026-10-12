# 青海大学

## 团队信息

- 提交者: wsnidie1
- 身份: 学生
- 单位: 青海大学

## 成员

- wsnidie1 (wsnidie1): 提交者

## 算子: challenge07_sigmoidcustom

8 核并行切分：host 侧 Tiling 将 16384 个元素按核均分为每核 2048 元素（向上取整，非整除无尾部丢弃），UB 单块搬运、含容量校验；kernel 侧用 AscendC 基础指令链 Muls→Exp→Adds→Div 实现 y=1/(1+exp(-x))，中间精度 fp32（fp16 走 Cast 往返），DataCopyPad 统一搬运；含越界核空转与尾核截断防御。

## 实现说明

- Tiling：8 核并行切分（向上取整、尾核截断、越界核空转防御），UB 单块搬运含容量校验
- Kernel：Muls→Exp→Adds→Div 指令链实现 y=1/(1+exp(-x))，中间精度 fp32，DataCopyPad 统一搬运
