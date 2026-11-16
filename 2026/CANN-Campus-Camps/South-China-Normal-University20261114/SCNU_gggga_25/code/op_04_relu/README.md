# op_04_relu

提交团队: 华南师范大学
提交者: gggga5


## 算子实现介绍

【Tiling策略】
针对大小Shape动态切分：小Shape（8x2048）限制8核并切分快发分支；大Shape（2048x2048）扩展至32核。BlockFactor按128对齐，UB Factor限制最大8192，兼顾负载均衡与内存约束。

【Kernel双分支设计】

    小Shape直通：针对16384元素极简场景，直接单次搬运与计算，裁剪冗余控制开销。

    大Shape流水线：手工编排DoubleBuffer（Ping-Pong）机制。申请 x[2] 和 y[2] 双缓冲，通过 Set/WaitFlag 精细控制 MTE2(搬入)、V(计算) 与 MTE3(搬出) 的并行，利用 V_MTE2 / MTE3_V 等信号实现搬运与计算流水掩盖。

【内存与指令】
调用原生 AscendC::Relu 高阶指令。全程采用 VECCALC 硬编码字节偏移划分无重叠UB空间（含STRIDE对齐），实现零动态内存分配；结合TilingKey做编译期模板派发。
