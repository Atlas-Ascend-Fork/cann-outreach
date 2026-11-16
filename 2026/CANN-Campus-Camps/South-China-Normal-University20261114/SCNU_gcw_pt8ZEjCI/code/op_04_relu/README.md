# op_04_relu

提交团队: 华南师范大学
提交者: gcw_pt8ZEjCI


## 算子实现介绍

本算子采用双缓冲流水线与动态多核切分策略。Tiling 阶段按 4KB（内存页边界）对齐切分 ubFactor 与 blockFactor，以最大化 DMA 搬运效率，同时针对小 Shape 动态按需开核以降低启动开销。Kernel 侧计算逻辑使用底层单指令 Maxs 替代通用接口；搬运时主数据块走极速 DataCopy，仅在尾部非对齐处使用 DataCopyPad，兼顾性能与越界安全。
