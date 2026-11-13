# op_04_relu

提交团队: 华南师范大学
提交者: 黄明沐


## 算子实现介绍

使用多核（AIV）实现，totalNum按blockFactor在核间均分，每核起始偏移32B对齐
UB切分：输入/输出队列各2份buffer（双缓冲），ubFactor=min(UB容量上限, blockFactor)，按32B对齐
核内流水：提前一个tile下发搬入，MTE2/V/MTE3跨tile重叠；常规tile走DataCopy，仅尾块用DataCopyPad
Tiling策略：blockFactor对齐到BLOCK_SIZE(32B)的整数倍，不申请workspace（WS=0）
