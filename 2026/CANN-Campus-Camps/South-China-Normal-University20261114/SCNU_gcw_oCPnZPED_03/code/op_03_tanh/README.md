# op_03_tanh

提交团队: 华南师范大学
提交者: gcw_oCPnZPED


## 算子实现介绍

Kernel 侧：多核实现，单缓冲。
Init 中通过 GetBlockIdx 定位本核负责的数据段起点（核号 × blockFactor），并按 totalNum 裁剪出
本核实际处理长度 blockLength，再取 ubLength = min(ubFactor, blockLength)；空核直接返回，不
申请 UB。使用一个 VECIN 队列与一个 VECOUT 队列（BUFFER_NUM = 1，每核一轮搬运即可完成，
单缓冲省去多余的流水同步开销）。Process 按 ubLength 分块循环，依次 CopyIn → Compute →
CopyOut：MTE2 将 x 从 GM 搬入 UB，向量计算后由 MTE3 写回 GM。

Compute 侧不使用 tanh 高阶 API，改用小算子拼接实现 tanh(x) = 1 - 2/(exp(2x)+1)：
Muls(2x) → Exp → Adds(+1) → Duplicate(常数 2) → Div → Muls(-1) → Adds(+1)。Float16 输入先
Cast 到 float32 参与计算、结果再 Cast(CAST_RINT) 回 half，保证半精度下的精度要求。

Tiling 策略：totalLength 按 BLOCK_SIZE(32B) 对齐，对齐粒度 alignNum = 32/sizeof(dtype)；
按“每核至少 2048 个元素”计算启动核数（16384 元素即 8 个核，每核 2048，恰好一行），
blockFactor 为每核元素数并对齐到 32B；ubFactor 由 UB 容量反推（按输入输出各一份 + float
临时两份估算）且不超过 blockFactor，保证每核一轮搬完。workspace 置 0，dtype 通过
tilingKey 在编译期分发（float16 → schMode 0，float32 → schMode 1），kernel 侧无分支判断。
