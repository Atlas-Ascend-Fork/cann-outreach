# op_04_relu

提交团队: 华南师范大学
提交者: ytle


## 算子实现介绍

多核切分：按 AIV 核数把 totalNum 均分到各核，每个核处理 blockFactor 个元素；最后一个核按剩余元素量收缩。
UB 切分：ubFactor 按 UB 容量和 4 块缓冲（输入、输出各 2 块）计算，并按 32B 对齐。
计算逻辑：y = max(0, x)，用 AscendC::Maxs 按标量比较实现；fp32 和 fp16 分别对应 tilingKey 1 和 0。
DoubleBuffer：以 CopyIn、Compute、CopyOut 三段式循环，BUFFER_NUM = 2 让 MTE2、V、MTE3 重叠执行。
Tiling 策略：totalNum 对齐到 BLOCK_SIZE（32B）的整数倍后再均分；blockFactor、ubFactor 均按 32B 对齐，保证 DataCopy 对齐访问，最后一个不完整的数据块由 kernel 按剩余量处理。
