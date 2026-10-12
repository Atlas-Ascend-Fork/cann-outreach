# 北京四维图新科技股份有限公司

## 团队信息

- 提交者: QZRuoCheng
- 身份: 企业员工
- 单位: 北京四维图新科技股份有限公司

## 成员

- QZRuoCheng (QZRuoCheng): 提交者

## 算子: challenge06_relucustom

基于 Ascend C 完成了 ReLU 算子的开源仓规范化实现：补全了 op_kernel/relu.h 的三段式流水（含末尾核截断与尾块处理）、op_host 的形状推导与多核+UB 两级切分 Tiling，并同步更新 UT 期望值为 92160 1440 1440，本地 stub 仿真验证全部通过。
