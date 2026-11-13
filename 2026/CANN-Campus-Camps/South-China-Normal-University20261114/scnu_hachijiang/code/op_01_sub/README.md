# op_01_sub

提交团队: 华南师范大学(石牌校区)
提交者: hachijiang


## 算子实现介绍

1. 8 核均分：blockLength = totalLength / 8 = 2048，各核用 SetGlobalBuffer 偏移切分，区间不重叠。
2. 核内 tileNum=8、双缓冲 BUFFER_NUM=2，tileLength=128 元素（32B 对齐），共 16 轮流水。
3. 搬运用 TQue + DataCopy，计算用 AscendC::Sub，KernelSub&lt;T&gt; 按 dtype 实例化 float/half。
