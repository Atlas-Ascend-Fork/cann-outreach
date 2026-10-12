# 青海大学

## 团队信息

- 提交者: wsnidie1
- 身份: 学生
- 单位: 青海大学

## 成员

- wsnidie1 (wsnidie1): 提交者

## 算子: challenge06_relucustom

本实现采用 8 核并行切分：host 侧 Tiling 将 16384 个元素按核均分为每核 2048 元素，单核单块、UB 按 32B 对齐一次搬入；kernel 侧使用 AscendC::Relu（vrelu 硬件矢量指令）单指令完成整块 max(0,x) 计算，经 DataCopy 流水（CopyIn→Compute→CopyOut）回写，无需 workspace。对非整除场景增加越界防御与尾部余数处理，保证任意 shape 下正确性。

## 实现说明

- Tiling：向上取整 8 核切分，非整除输入由尾核负责余数元素（无尾部丢失），越界核空转防御；UB 单块搬运含容量校验
- Kernel：`AscendC::Relu` 向量指令逐元素计算，DataCopyPad 统一搬运
- 修复记录：首版 blockFactor 向下取整导致非整除输入尾部丢失，已改为向上取整 + 尾核截断 + 混合数据逐元素断言测试
