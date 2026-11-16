# op_02_mul

提交团队: 华南师范大学
提交者: xly0411


## 算子实现介绍

使用Ascend C矢量编程范式实现矢量乘法算子z = x * y，shape (8, 2048)，支持float16/float32（TilingKey按输入dtype自动分发模板实例）。

并行与Tiling策略：host侧TilingFunc根据输入shape计算totalLength=16384，SetBlockDim(8)启动8个AI Core做SPMD数据并行，每核处理blockLength=2048个元素；片内按tileNum=8分块、BUFFER_NUM=2双缓冲，tileLength=128个元素（float32为512B、float16为256B，均为32B的整数倍，满足DataCopy对齐要求）。

流水设计：CopyIn通过DataCopy将两路输入从Global Memory搬入UB并以VECIN队列同步；Compute调用Mul高阶API完成逐元素相乘，结果入VECOUT队列；CopyOut将结果DataCopy写回Global Memory。三级流水配合双缓冲重叠执行，提升并行度。
