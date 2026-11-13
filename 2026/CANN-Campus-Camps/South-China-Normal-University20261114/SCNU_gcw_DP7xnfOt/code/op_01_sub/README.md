# op_01_sub

提交团队: 华南师范大学
提交者: 何建基


## 算子实现介绍

使用 Ascend C 多核并行实现矢量减法 z=x-y。Host 侧根据输入 shape 计算 totalLength，并按 blockDim=8 切分到各核；核内将 blockLength 按 tileNum 和 BUFFER_NUM 分块，利用 double buffer 通过 DataCopy 搬入 UB，调用 AscendC::Sub 完成减法后再搬出到 Global Memory。支持 float32 与 float16 数据类型。
