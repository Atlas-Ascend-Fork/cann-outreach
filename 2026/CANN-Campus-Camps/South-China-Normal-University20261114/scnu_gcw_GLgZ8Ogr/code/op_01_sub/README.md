# op_01_sub

提交团队: 华南师范大学
提交者: 吴靖飞


## 算子实现介绍

采用多核并行实现，blockDim=8，按 GetBlockIdx() 将数据均分到各核，每核处理 blockLength = totalLength / 8 个元素。单核内使用 TPipe + TQue 做 Double Buffer 流水（BUFFER_NUM=2），tileNum=8，tileLength = blockLength / tileNum / BUFFER_NUM，循环执行 CopyIn → Compute → CopyOut。Tiling 策略：totalLength 由输入 shape 连乘得到（8×2048=16384），tileNum 取 8；核函数按 dtype 选择 float/half 模板实例。计算调用 AscendC::Sub 完成 z = x - y。
