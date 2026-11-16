# op_01_sub

提交团队: 华南师范大学
提交者: camdies


## 算子实现介绍

## 1. 整体流程
按 多核并行 + 单核内分块流水 方式实现：host 侧 run_kernel 从 TensorInfo 解析出总元素数 totalLength（8×2048）与数据类型 dtype，按 dtype 选择 sub_custom&lt;float&gt;（fp32）或 sub_custom&lt;half&gt;（fp16）实例，以 &lt;&lt;&lt;8, nullptr, stream&gt;&gt;&gt; 启动 8 个核。

## 2. Tiling 切分
每核承担 blockLength = totalLength / GetBlockNum() 个元素；核内再按 tileLength = blockLength / tileNum / BUFFER_NUM 切成若干数据块，配合 BUFFER_NUM=2 双缓冲实现搬入/计算/搬出的流水重叠。

## 3. 核心计算
CopyIn：从 Global Memory 经 DataCopy 将 x、y 数据块搬入 Local（UB）缓冲区，EnQue 入队；
Compute：DeQue 取数，调用矢量指令 AscendC::Sub(zLocal, xLocal, yLocal, tileLength) 完成 z = x − y，结果入输出队列；
CopyOut：DeQue 取出结果，DataCopy 写回 Global Memory 对应位置。
三类队列（TQue&lt;VECIN&gt;×2 + TQue&lt;VECOUT&gt;）配合 TPipe 统一管理片上内存，循环覆盖核内全部数据块。
