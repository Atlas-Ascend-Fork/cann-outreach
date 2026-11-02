# challenge11_sqrtcustom

提交团队: 个人开发者
提交者: weixin_48190404


## 算子实现介绍

采用多核及 UB 32B 对齐切分策略，totalLength 严格对齐到 BLOCK_SIZE 整数倍。Kernel 侧基于双缓存流水线设计，并使用 DataCopyPad 保障尾块安全搬运；针对 Half 类型采用 FP32 升精度计算（四舍五入回写），全面兼顾了高吞吐与 PyTorch 原生精度对齐。
