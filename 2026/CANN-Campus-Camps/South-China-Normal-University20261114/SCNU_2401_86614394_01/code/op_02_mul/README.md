# op_02_mul

提交团队: 华南师范大学
提交者: 王佳豪


## 算子实现介绍

实验2：Ascend C向量乘法自定义算子，z=x*y，支持(8,2048) ND float32/float16。补齐Host形状与类型推导、模板dtype选择及Tiling结构；最多8核均分，UB容量允许时单buffer整块处理，否则按二分缩小tile。Kernel采用DataCopy—Mul—DataCopy和完整队列生命周期。基础与优化实现通过平台全部2个验证点；提供代码、真实过程截图及优化对比报告。
