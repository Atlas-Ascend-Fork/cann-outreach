# op_02_mul

提交团队: 华南师范大学
提交者: 吴靖飞


## 算子实现介绍

采用多核并行（按平台 AIV 核数选择可整除 totalLength 的最大核数，本用例 blockDim=32），每核通过 GetBlockIdx() 计算 GM 偏移，处理 blockLength=totalLength/blockDim 个元素。单核内使用 TPipe+TQue 双缓冲流水，tileLength = blockLength / tileNum / BUFFER_NUM（tileNum=8、BUFFER_NUM=2），循环执行 CopyIn→Compute→CopyOut。Tiling 策略：totalLength 由输入 shape 连乘得到（8×2048=16384），tile 长度与 GM 偏移均保证 32B 对齐，不需要 DataCopyPad。计算调用 AscendC::Mul 完成 z=x*y，核函数按 dtype 选择 float/half 模板实例
