# op_01_sub

提交团队: 华南师范大学
提交者: gggga5


## 算子实现介绍

该算子实现了一个基于 Ascend C 的逐元素向量减法 z = x - y，核函数 lesson1_sub_fast&lt;T&gt; 通过模板支持 float 和 half 两种数据类型。运行时 run_kernel 根据输入张量 info_x.tensors[0].dtype 选择具体数据类型实例化核函数，并在 availableCoreNum &lt;= 0 时直接返回。多核切分采用固定总量 16384 元素的静态策略：blockNum 初始为 8，并根据可用核数 availableCoreNum 逐次折半，直到不超过可用核数；每个核处理 count = 16384 / blockNum 个连续元素，核内通过 GetBlockIdx() * count 计算全局内存偏移，从而将输入输出按核划分。

核函数内部使用 GlobalTensor 绑定 x、y、z 的全局内存切片，并在 VECCALC 位置的 TBuf 中一次性申请 3 * count * sizeof(T) 的 Local 空间，将同一块 LocalTensor 按 count 偏移划分为 xLocal、yLocal、zLocal。随后通过两次 DataCopy 将 x、y 从 GM 搬入 Local，并使用 MTE2_V 事件进行 Set/Wait 同步，确保搬入完成后才执行 Sub(zLocal, xLocal, yLocal, count) 向量计算；计算完成后再通过 V_MTE3 事件同步，最后将 zLocal 搬回全局内存 zGm。该实现是固定长度、单次搬入搬出的教学型向量减法样例，未处理动态 shape、尾块及 UB 容量自适应等场景。
