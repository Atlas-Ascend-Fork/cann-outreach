# challenge06_relucustom

提交团队: 电子科技大学
提交者: StudentXHF


## 算子实现介绍

使用多核分片和双缓冲队列处理输入；Host 根据 totalNum、AIV 核数和 UB 容量计算 blockFactor/ubFactor，Kernel 采用 CopyIn→AscendC::Relu→CopyOut 流水线，支持 float32/float16。
