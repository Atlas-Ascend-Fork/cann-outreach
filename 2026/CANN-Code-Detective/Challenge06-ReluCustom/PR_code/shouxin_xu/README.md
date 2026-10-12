# 北京首芯科技文化有限公司

## 团队信息

- 提交者: gang1022
- 身份: 企业员工
- 单位: 北京首芯科技文化有限公司

## 成员

- gang1022 (gang1022): 提交者

## 算子: challenge06_relucustom

基于 Ascend C 实现 ReLU 算子，支持 float16 和 float32。Tiling 侧采用 8 个 Vector Core，每核划分 16 个 tile；Kernel 完成 GM 到 UB 搬运、AscendC::Relu 计算及结果回写，workspace 为 0。CANNJudge 验证集全部 PASS。
