# op_04_relu

提交团队: 华南师范大学
提交者: gcw_oCPnZPED


## 算子实现介绍

Kernel 侧：多核 + 软件流水。Init 中记录全局元素总数、tile 长度与核数，空核直接返回不申请 UB；
所有核共用同一套全局 buffer，靠 tile 下标定位，不做 per-core 偏移。输入队列深度 3、输出队列
深度 2（BUFFER_NUM），Process 采用软件流水：先把前 2 块的搬入发出去，之后每轮先发第 k+2 块的
搬入，再计算第 k 块、搬出第 k 块，使 MTE2 在计算与搬出的同时持续搬运，避免搬运单元空转。
分块采用交错（循环）分块：核 b 处理第 b、b+核数、b+2*核数… 个 tile，使所有核在同一时刻读写
相邻的一小片地址窗口，DRAM 行局部性与 L2 命中率优于每核各占一段远区间。

Compute 侧：ReLU 即 max(x, 0)，直接调用一条 Maxs(y, x, 0, n) 完成整块逐元素计算，不引入
float 中间 buffer、不做 Cast（输入输出同 dtype），也不使用 ReLU 高阶 API。

Tiling 策略：totalLength 按 BLOCK_SIZE(32B) 对齐，并对齐粒度取 512B（alignNum=32/sizeof(dtype)）；
按“每核至少 2048 个元素”计算启动核数（16384 元素启动 8 核、每核一块；4194304 元素直接满核）；
tile 长度由“每核平均量”和 UB 容量共同决定，UB 只用 3/4，按 (IN_BUFFER_NUM + OUT_BUFFER_NUM)
份 buffer 反推上限并 512B 对齐，实现中 fp32 为 7296 元素（28.5KB）、每核 15 块；workspace 置 0，
dtype 经 tilingKey 在编译期分发（float16→schMode 0，float32→schMode 1）。
