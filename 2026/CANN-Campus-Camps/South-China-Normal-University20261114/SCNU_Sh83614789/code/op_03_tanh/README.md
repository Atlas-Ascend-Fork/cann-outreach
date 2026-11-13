# op_03_tanh

提交团队: 华南师范大学
提交者: Sh83614789


## 算子实现介绍

本算子基于 Ascend C 实现 y = tanh(x)，输入 x、输出 y 形状均为 (8, 2048)，共 16384 个元素，支持 float32 / float16，数据排布为 ND。

一、多核切分
Tiling 阶段由输入 shape 逐维相乘得到总元素数 totalLength = 16384，按 8 个 AIV 核均分，通过 SetBlockDim(8) 下发到 kernel 侧，每核处理 2048 个元素；kernel 侧用 GetBlockIdx() 计算本核在 Global Memory 上的偏移 blockLength * blockIdx，各核处理区间互不交叠，实现多核并行。

二、片内切块与双缓冲
每核 2048 个元素再按 tileNum = 8、BUFFER_NUM = 2 切分，单块处理 128 个元素，float 对应 512B、float16 对应 256B，均满足 32 字节对齐要求，单核循环 16 次完成全部数据处理。输入队列与输出队列各申请 2 块 UB 缓冲，构成双缓冲。

三、流水设计
采用 CopyIn → Compute → CopyOut 三级流水：DataCopy 从 Global Memory 搬入 UB，矢量指令在 UB 上完成 tanh 计算，再搬回 Global Memory。双缓冲使搬运与计算在时间上重叠，减少搬运等待。

四、计算实现
按 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) 实现：先 Exp 得到 exp(x)，再经 Muls(-1) 后 Exp 得到 exp(-x)；Add 得到分母 exp(x) + exp(-x)，Sub 得到分子 exp(x) - exp(-x)，最后用精确除法 Div 得到结果。相比使用近似倒数指令，Div 的误差更小，能稳定满足判题精度要求。中间结果使用 UB 上 4 块常驻临时 buffer 存放，避免重复申请。

五、精度处理
float16 输入先通过 Cast 转为 float32，在 float32 上完成 exp、相减、相除的全部运算，再通过 Cast 转回 float16 输出；float32 输入直接在 float32 上计算。这样规避了 float16 尾数位不足导致分子差值有效位丢失的问题。

六、内存与同步
输入、输出队列通过 AllocTensor / EnQue / DeQue / FreeTensor 完成 UB 内存的申请、同步与释放，计算完成后及时 FreeTensor 回收，保证 UB 复用；临时缓存使用 TBuf 常驻，一次申请全程复用。
