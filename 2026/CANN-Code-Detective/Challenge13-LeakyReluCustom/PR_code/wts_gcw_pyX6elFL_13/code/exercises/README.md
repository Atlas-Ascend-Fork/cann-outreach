# exercises

提交团队: 梧桐森智能科技（东莞）有限公司
提交者: sws168


## 算子实现介绍

从 Host 读取 negative_slope 并通过 TilingData 下发。按 32 字节对齐进行多核切分和 UB 搬运；float32 直接执行向量 LeakyReLU，float16 转为 float32 计算后以就近偶数舍入写回。CANNJudge 提交 1247975，2/2 Pass。
