# op_01_sub

提交团队: 华南师范大学
提交者: 刘子清


## 算子实现介绍

本算子基于 Ascend C 实现矢量减法 z=x-y。Host 侧根据总元素数和 tileNum 生成 tiling，并将数据按 8 个核均分；核内再按 tileNum 和 BUFFER_NUM 分块，通过 TPipe 与 VECIN/VECOUT 队列管理 UB，用 DataCopy 搬入 x、y，调用 AscendC::Sub 计算后搬出 z，支持 fp32/fp16 模板实例。
