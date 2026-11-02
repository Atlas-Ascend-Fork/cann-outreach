# op_01_sub

提交团队: 华为技术有限公司
提交者: LikeWindc


## 算子实现介绍

多核连续分段 + 核内 16KB 双缓冲流水；Tiling 策略为 numBlocks = min(核数, ceil(totalLength/1024))、每核 blockLength = ceil(totalLength/numBlocks)
