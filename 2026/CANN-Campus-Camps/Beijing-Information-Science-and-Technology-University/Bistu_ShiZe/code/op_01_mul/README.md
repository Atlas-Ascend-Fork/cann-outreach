# op_01_mul

提交团队: 北京信息科技大学
提交者: 师泽


## 算子实现介绍

采用 8核并行 实现，基于 Vector 计算单元执行逐元素乘法

Tiling策略：totalLength 均分至8核，单核内按 tileNum(8) × BUFFER_NUM(2) 进一步切分，单块长度 tileLength = blockLength / 16

内存管理：基于 TPipe 和 TQue 队列管理 UB 内存，输入输出队列挂载于 VECIN/VECOUT

流水线：CopyIn → Compute → CopyOut 严格串行执行，未启用双缓冲重叠流水

类型支持：采用模板化编程，根据 Host 侧 dtype 分派 float (fp32) 与 half (fp16) 实例
