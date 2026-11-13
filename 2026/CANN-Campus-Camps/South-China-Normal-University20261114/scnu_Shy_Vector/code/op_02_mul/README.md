# op_02_mul

提交团队: 华南师范大学
提交者: Shy_Vector


## 算子实现介绍

一、算子功能
对两个形状均为 (8, 2048) 的输入张量执行逐元素乘法 z = x * y，支持 float32 / float16，ND 格式。

二、工程结构
文件	作用
op_host/mul.cpp	Host 侧：算子定义、InferShape/InferDataType、Tiling 计算
op_kernel/mul.cpp	Kernel 侧：KernelMul 类，执行 搬运→计算→回写 流水
op_kernel/mul_tiling.h	Tiling 数据结构（length、block_factor）
op_kernel/tiling_key_mul.h	模板参数声明（按数据类型 DT_X 区分 fp32/fp16，编译期生成两份 kernel 二进制）
三、Host 侧实现（op_host/mul.cpp）
平台信息：PlatformAscendC 获取 AIV 核数 num_cores_aiv 与 UB 容量。
Tiling 计算（性能优化点）：
length_x = 16384（8×2048）；
以「每核约 8KB 数据量」为目标反推核数 block_dim：fp32 → 8 核，fp16 → 4 核；
block_factor = 每核元素数，向上对齐到 32 字节（align_elems）；
若超过 UB 容量（ub_size/4）则裁剪并增加核数兜底，保证大 shape 通用性。
Tiling 数据下发：写入 MulTilingData（length、block_factor），SetBlockDim(block_dim)。
TilingKey：ASCENDC_TPL_SEL_PARAM 按输入 dtype 选择对应模板实例。
InferShape/InferDataType：输出 z 的形状、类型与 x 一致。
四、Kernel 侧实现（op_kernel/mul.cpp）
单核处理流程（Process）：


start = block_id * block_factor            // 本核起始偏移
core_len = min(block_factor, length-start) // 本核实际元素数
CopyIn(start, core_len)   // GM→UB：x、y 各一次 DataCopy（32B对齐段）
Compute(core_len)         // UB 内 Mul：z = x * y
CopyOut(start, core_len)  // UB→GM：z 一次 DataCopy
队列：TQue&lt;VECIN,1&gt; 两条（x/y）+ TQue&lt;VECOUT,1&gt; 一条，深度 1；
尾块处理：非 32B 对齐的尾巴用 DataCopyPad 补齐搬运（本 shape 下整除、路径不触发）；
TQue 的 EnQue/DeQue 自动插入 MTE↔VECTOR 同步事件，无需手动 SetFlag/WaitFlag。
五、优化要点（相对原始模板）
原始模板用满全部 40 核、每核仅 ~416 元素。实测 msprof 显示向量计算仅 0.04μs、搬运 ~1.3μs，而每核固定标量/调度开销约 1.4μs（占 57%），核数越多同步波开销越大。

优化后改为适度核数（每核 ~8KB，fp32 8 核 / fp16 4 核），核时间由约 3.9μs 降至 fp32 ~1.9μs / fp16 ~1.7μs，提升约 2 倍；block_factor 仍保持 32 字节对齐，尾部 DataCopyPad 逻辑保留以保证任意 shape 正确性。

如需进一步压榨，可考虑去掉 DataCopyPad 尾处理分支（本 shape 不触发，收益有限），或改为 TBuf 直连（实测与 TQue 性能持平）。
