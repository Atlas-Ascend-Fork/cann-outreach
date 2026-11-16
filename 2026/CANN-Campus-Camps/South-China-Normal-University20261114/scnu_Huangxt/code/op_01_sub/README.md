# op_01_sub

提交团队: 华南师范大学
提交者: 黄熙桐


## 算子实现介绍

基于AscendC模板实现向量减法z=x-y，沿用TPipe、TQue分块搬运逻辑，仅将计算算子替换为AscendC::Sub，支持float32/float16。
