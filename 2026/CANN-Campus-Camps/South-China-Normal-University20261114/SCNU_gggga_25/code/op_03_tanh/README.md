# op_03_tanh

提交团队: 华南师范大学
提交者: gggga5


## 算子实现介绍

Tiling策略：针对固定Shape(8, 2048)，固定8核并行，每核精准处理2048个元素，无尾块判断开销。
计算优化：采用 2/(1+exp(-2x))-1 公式变形，将Exp指令调用减半；FP16/BF16输入统一Cast升FP32计算，保障精度与PyTorch对齐。
内存与流水：手动按32B对齐规划UB空间（隔离x/a/b/y），通过PipeBarrier等严格保证流水线同步。
模板派发：结合TilingKey传入schMode，在Kernel侧利用constexpr实现不同dtype的编译期分支消除。
