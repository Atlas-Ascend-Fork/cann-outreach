# op_01_sub

提交团队: 上海理工大学
提交者: kylin0326


## 算子实现介绍

本算子基于 Ascend C 实现矢量减法功能（z = x - y），支持 float32 和 float16 两种数据类型。

【实现方案】
1. Tiling策略：Host 侧推导总元素个数 totalLength。采用多核并行（blockDim=8），由 AscendC::GetBlockNum() 和 GetBlockIdx() 完成多核数据切分。
2. 核内切分：单核数据按 tileNum=8 进行分块，并结合 BUFFER_NUM=2 采用双缓冲（Double Buffer）机制，实现 CopyIn -&gt; Compute -&gt; CopyOut 的流水线并行。
3. UB 切分与对齐：依据 tileLength * sizeof(T) 申请 UB 空间，自动满足 32 字节对齐要求。
4. 计算逻辑：使用 AscendC::Sub 接口完成逐元素相减，并在计算完成后及时调用 FreeTensor 释放内存。
