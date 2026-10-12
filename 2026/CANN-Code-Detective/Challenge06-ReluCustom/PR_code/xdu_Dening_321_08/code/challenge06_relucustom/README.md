# challenge06_relucustom

提交团队: 西安电子科技大学
提交者: Dening_321


## 算子实现介绍

ReLU 算子：8 核并行，每核 11520 个元素；UB 双缓冲 16B/元素，一趟搬完；核心计算一条 Maxs(y, x, 0) 指令完成。
