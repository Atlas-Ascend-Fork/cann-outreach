# exercises

提交团队: 个人开发者
提交者: Code_demo111111


## 算子实现介绍

本算子基于Ascend C多核架构实现LeakyReLU激活计算：Host侧完成输入切分与斜率属性解析，经TilingData传递给Kernel；Kernel侧各核独立处理数据分片，采用双缓冲流水搬运，正负区间分段计算（正数直通、负数乘以斜率），支持half与float两种精度。
