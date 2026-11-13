# op_01_sub

提交团队: 华南师范大学
提交者: Shy_Vector


## 算子实现介绍

一、功能与接口
功能：逐元素计算 z = x - y，输入输出形状均为 (8, 2048)，支持 float32 / float16，数据格式 ND。
接口：extern "C" void run_kernel(...) 为 host 侧启动入口（kernel.asc:81），sub_custom&lt;T&gt; 为 Device 核函数（kernel.asc:74），按 dtype 分别实例化 float / half 模板。
规模：总元素 16384；fp32 每张量 64KB，fp16 每张量 32KB，属极小 data-size、延迟敏感型算子。
二、整体结构
采用「Host 侧预计算 Tiling + Device 侧单核单 Tile」的直接调用模式：

Host 侧（run_kernel）

由 TensorInfo 推导 totalLength 与 dtype；
计算 blockNum、perBlock，打包成 SubTilingData{totalLength, perBlock} 按值传入核函数，避免 Device 侧重复做除法。
Device 侧（KernelSub）

Init：根据 GetBlockIdx() 算出本核在 GM 上的起始偏移 start 与长度 len_，用 SetGlobalBuffer 绑定 x/y/z 的分片；用 TBuf 申请 UB 缓冲。
Process：一次 DataCopy 搬入 → Sub 计算 → DataCopy 搬出。
三、关键实现设计
1. 分核策略（4 核，4096 元素/核）

elemPerCore = 4096;
blockNum   = ceil(totalLength / elemPerCore);   // 本 shape → 4 核
每核处理 4096 元素（fp32 16KB / fp16 8KB），TILE_LENGTH=4096 保证 perBlock ≤ TILE_LENGTH，即每核只处理一个 Tile，无循环。
实测核数与耗时呈 U 形：核太少单核串行长、核太多核间调度/同步开销大，4 核为最优点。
2. 用 TBuf 替代 TQue（核心优化）
原实现走 TQue 的 AllocTensor/EnQue/DeQue/FreeTensor 全流程。profiling 显示其标量流水线占用 76%（队列索引维护 + 每次跨界事件同步），而矢量计算仅占 ~9%。改用裸 TBuf 只做缓冲申请，去掉了队列管理开销，标量占用显著下降。

3. 跨流水线同步策略

DataCopy(x, ...); DataCopy(y, ...);
PipeBarrier&lt;PIPE_ALL&gt;();   // MTE2 -&gt; V：确保搬入完成再计算
Sub(...);
PipeBarrier&lt;PIPE_V&gt;();     // V -&gt; MTE3：确保计算完成再搬出
DataCopy(z, ...);          // 末尾无需屏障，任务结束保证 MTE3 排空
注意：MTE2→V 必须用 PIPE_ALL。仅用 PipeBarrier&lt;PIPE_MTE2&gt;() 不能建立跨流水线依赖，Sub 会读到上次启动的残留 UB 数据（循环压测时因 buffer 已有正确值而掩盖，单次运行必现错误）。
4. 对齐处理
perBlock 按 dtype 向上对齐到 32B 粒度（fp32:8 元素、fp16:16 元素），起始偏移与长度均 32B 对齐，全程走高效 DataCopy，无需 DataCopyPad 慢路径。
四、性能结果
msprof task_time 稳态（min / median）：

数据类型	原实现	优化后	榜单 TBest
fp32	1.70 / 1.88 μs	1.58–1.60 / 1.70 μs	1.76 μs
fp16	1.66 / 1.80 μs	1.46 / 1.58 μs	1.68 μs
正确性：fp32/fp16 单次运行均与 golden 精确一致（Max diff = 0.0），run.sh 验证 PASS。
