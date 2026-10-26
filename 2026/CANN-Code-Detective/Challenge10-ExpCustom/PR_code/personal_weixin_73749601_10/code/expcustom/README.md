# expcustom

提交团队: 个人
提交者: Cola_Coder


## 算子实现介绍

多核切分实现：tiling 按 MIN_SPLIT_THRESHOLD=2048 控制每核最少数据量，shape 各维累乘得到 totalNum，blockFactor = CeilDiv(totalNum, usedCoreNum)，ubFactor 按 UB 双队列×双缓冲并预留一半 UB 给高阶 API 临时空间来预算，SetBlockDim 按实际所需核数启动。kernel 采用 CopyIn→Compute→CopyOut 三段式流水，实际分块 cap 到 4096 元素并向上对齐到 32B，对齐走 DataCopy 快路径，非对齐尾部回退 DataCopyPad 保证任意形状正确性；AscendC::Exp 逐元素计算，BUFFER_NUM=2 双缓冲掩盖搬运耗时。CANNJudge 验证集 PASS，当前榜首（2.27s/2.19s，precision_ratio=1）。
