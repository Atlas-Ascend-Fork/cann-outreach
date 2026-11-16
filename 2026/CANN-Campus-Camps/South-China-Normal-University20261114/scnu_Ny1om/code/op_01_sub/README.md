# op_01_sub

提交团队: 华南师范大学
提交者: 刘星悦


## 算子实现介绍

单核实现，totalLength 按 32B 对齐，核内 tileNum 切块，tileLength 保持 32B 对齐，三级流水 + 双缓冲（BUFFER_NUM=2）重叠搬运与计算，模板化支持 fp32/fp16。
