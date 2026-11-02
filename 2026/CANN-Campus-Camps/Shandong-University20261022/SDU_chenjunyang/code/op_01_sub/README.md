# op_01_sub

提交团队: 山东大学
提交者: 陈君扬


## 算子实现介绍

本算子基于 Ascend C Direct Invocation 实现矢量减法 z = x - y：Host 侧由 run_kernel 根据输入张量的 shape 计算总元素数，结合 availableCoreNum 将任务按 16 个元素对齐后均匀划分到多个 Vector Core，并通过 SubCustomTilingData 向 Kernel 传递总长度、单核长度、UB 分片长度和数据类型；Device 侧由模板类 KernelSub&lt;T&gt; 同时支持 float32 和 float16，每个核通过 GetBlockIdx() 确定自身负责的全局偏移，将数据按 4096 个元素分片，使用 TPipe、TQue 的 VECIN/VECOUT 双缓冲依次完成 DataCopy 搬入、Sub 逐元素相减和 DataCopy 搬出，并通过尾块长度判断保证最后一个核及最后一个分片不会越界；入口函数 sub_custom 根据 Tiling 中的 dtype 分派到对应的数据类型实例，从而在多核并行和搬运计算流水化的基础上正确、高效地完成矢量减法。
