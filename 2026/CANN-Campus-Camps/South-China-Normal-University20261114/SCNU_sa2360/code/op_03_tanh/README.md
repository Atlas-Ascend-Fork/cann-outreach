# op_03_tanh

提交团队: 华南师范大学
提交者: sa2360


## 算子实现介绍

Ascend C Tanh 采用单核分块计算，按 ubFactor 将输入分段搬入 UB，使用 Exp、Muls、Sub、Add、Div 按 tanh(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x)) 计算，再写回输出。Tiling 根据输入形状计算 totalNum，并结合 UB 容量设置 ubFactor；分块长度按 32B 对齐，末块按实际长度处理。本题 totalNum=16384，支持 float32 和 float16。
