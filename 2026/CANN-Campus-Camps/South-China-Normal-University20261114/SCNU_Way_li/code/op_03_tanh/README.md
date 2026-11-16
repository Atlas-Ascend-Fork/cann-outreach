# op_03_tanh

提交团队: 华南师范大学
提交者: Way_li


## 算子实现介绍

算子基于 Ascend C 实现自定义激活算子 Tanh，支持 Float16、Float32 输入输出。Host 侧完成形状推导和 Tiling，将数据分配至多个核；Kernel 侧采用双缓冲，通过指数、加法、乘法、除法等小算子拼接实现 Tanh，不直接调用内置 Tanh API。FP16 数据使用 FP32 中间计算以提高精度，并提供 aclnn 调用测试代码，用于校验输出结果
