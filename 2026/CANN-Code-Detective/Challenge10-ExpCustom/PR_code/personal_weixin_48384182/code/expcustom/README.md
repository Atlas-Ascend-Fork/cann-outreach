# expcustom

提交团队: 个人
提交者: weixin_48384182


## 算子实现介绍

多核 + UB 两级切分：每核 blockFactor=1024（16 核覆盖 16384），UB 内 ubFactor 由容量反推并 32B 对齐；totalLength 按 32B 对齐，按 dtype 选 tilingKey；Kernel 走 CopyIn→Compute→CopyOut，双缓冲，AscendC::Exp 计算。
