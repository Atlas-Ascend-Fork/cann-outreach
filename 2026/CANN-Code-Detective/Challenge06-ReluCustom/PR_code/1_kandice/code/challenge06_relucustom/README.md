# challenge06_relucustom

提交团队: 1
提交者: kandice


## 算子实现介绍

多核(8 核)实现、CopyIn → Relu → CopyOut 流水、UB 切分按 32B 对齐。Tiling 策略：totalLength 对齐到 BLOCK_SIZE 整数倍，usedCoreNum 尾核自动减核，tileLength/tileNum 计算方式及尾块钳制。
