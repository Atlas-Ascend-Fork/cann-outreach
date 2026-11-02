# challenge11_sqrtcustom

提交团队: 个人开发者
提交者: Code_demo111111


## 算子实现介绍

Tiling 策略：多核均匀切分，totalLength 及 UB 切分严格对齐到 32B（BLOCK_SIZE）整数倍。
Kernel 实现：采用双缓存流水线，基于 DataCopyPad 处理尾块防越界；针对 FP16/BF16 采用升精度至 FP32 计算并四舍五入回写，确保与 PyTorch 精度严格对齐。
