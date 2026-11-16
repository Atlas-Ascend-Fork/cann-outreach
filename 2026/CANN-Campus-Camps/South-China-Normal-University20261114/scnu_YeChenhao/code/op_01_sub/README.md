# op_01_sub

提交团队: 华南师范大学
提交者: gcw_Zqa1u7nr


## 算子实现介绍

本算子基于 CANN 9.0.0 的 Ascend C 算子核函数工程，实现逐元素矢量减法 z = x - y，输入输出均为 (8, 2048) 的 ND 张量，支持 float32 与 float16。工程中唯一需要编写的文件是 kernel.asc，它由 main.asc 通过 #include 引入，整体由四部分组成：host 侧计算并传入的切分参数结构体 SubCustomTilingData、模板化的核函数主体类 KernelSub&lt;T&gt;、global vector 核函数入口 sub_custom，以及 extern "C" 的启动函数 run_kernel。run_kernel 从 TensorGroupInfo 中读取形状与 dtype，dtype 为 1 时实例化 half 版本，否则使用 float 版本，再按 &lt;&lt;&lt;blockDim, nullptr, stream&gt;&gt;&gt; 启动多核。数据流为 GM 经 DataCopy 搬入 UB 的 LocalTensor，调用 AscendC::Sub 完成 dst = src0 - src1，结果再经 DataCopy 搬回 GM；多核部分把 16384 个元素按 32 字节对齐块均匀分给各 AI Core，前 extra 个核多分一块，保证核数不整除时也不漏算、不越界。性能上，由于数据总量仅 192 KB，瓶颈在指令调度而非带宽，因此把切分运算全部放在 host 侧使 kernel 内零除法，每核只做一次大块搬入、一次 Sub、一次搬出，用 TBuf 取代 TQue 省去队列同步，输出复用输入 buffer 少占一块 UB，并用 SetFlag/WaitFlag 精确同步 MTE2→V→MTE3 三条流水。评测以 float64 计算的 x - y 为 golden，按 rtol=1e-4、atol=0.1 逐元素比对，得分由 kernel 用时与全场最快用时的对数比值决定，因此减少指令与同步次数的优化能直接体现在成绩上。
