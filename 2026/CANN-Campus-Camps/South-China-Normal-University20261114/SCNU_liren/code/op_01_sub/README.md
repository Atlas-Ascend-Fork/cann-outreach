# op_01_sub

提交团队: 华南师范大学
提交者: 李俊杰


## 算子实现介绍

采用「8核并行 + 核内二次分块 + 双缓冲」的矢量实现。
Host 侧从 TensorGroupInfo 连乘得到 totalLength=16384，tileNum 固定为 8。
多核切分：blockLength = totalLength / GetBlockNum() = 2048，每个核按
blockLength*GetBlockIdx() 定位 GM 起点，8 核恰好铺满 16384 元素。
核内再按 tileNum×BUFFER_NUM 分块：tileLength = 2048/8/2 = 128，循环 16 次。
UB 为 x/y/z 各分配 2 块，单块 128 元素（fp32 512B、fp16 256B），均满足 32B 对齐。
流水为 DataCopy 搬入 → AscendC::Sub 计算 → DataCopy 搬出，队列深度 1、双缓冲复用。
核函数模板化，按输入 dtype 分派 float / half 两个实例。
