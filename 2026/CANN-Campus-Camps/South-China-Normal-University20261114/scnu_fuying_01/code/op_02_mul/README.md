# op_02_mul

提交团队: 华南师范大学
提交者: gcw_vuifW5gj


## 算子实现介绍

采用 Host 侧 Tiling + Kernel 侧核函数两段式结构。
Tiling 将总长度向上对齐到 BLOCK_SIZE(32B) 整数倍，满足 DataCopy 搬运对齐要求。
通过 SetBlockDim(8) 启动 8 个 AIV 核并行拆分任务。
单核内按 Tile 切分并采用 BUFFER_NUM=2 双缓冲流水。
循环执行 CopyIn（GM→UB）、Compute（AscendC::Mul 矢量乘法）、CopyOut（UB→GM）三段流水。
全程由 TPipe 管理 UB Buffer，TQue&lt;VECIN&gt;/&lt;VECOUT&gt; 队列配合 EnQue/DeQue 完成 MTE2↔V↔MTE3 流水同步。
数据类型多态通过 ASCENDC_TPL 模板参数 DT_X 声明，经 ASCENDC_TPL_SEL_PARAM 在 Host 侧选取 TilingKey 实例化对应 Kernel 模板。
Kernel 侧以 template&lt;class DT_X&gt; 特化，由 AscendC::Mul 自动选择 fp16/fp32 矢量乘指令。
在加法算子框架基础上仅将 AscendC::Add 替换为 AscendC::Mul 并相应更名。
分块、双缓冲、多核并行、32B 对齐与 dtype 模板分发等逻辑完整复用，正确高效实现 z = x * y。
