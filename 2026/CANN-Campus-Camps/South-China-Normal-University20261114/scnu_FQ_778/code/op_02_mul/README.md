# op_02_mul

提交团队: 华南师范大学
提交者: FQ_778


## 算子实现介绍

按评审意见修改代码，输入描述改为从 GetInputDesc 取，判空，totalLength 用 int64 并加上限校验，核间与核内余数都由 kernel 处理，并采取小 shape 保护，tileLength 由 host 按 UB 预算自适应，搬运改用 DataCopyPad 以消除 32B 对齐问题。
