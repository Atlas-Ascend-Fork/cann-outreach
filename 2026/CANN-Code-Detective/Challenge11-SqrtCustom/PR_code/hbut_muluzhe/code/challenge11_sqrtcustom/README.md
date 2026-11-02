# challenge11_sqrtcustom

提交团队: 湖北工业大学
提交者: muluzhe


## 算子实现介绍

本方案基于 Ascend C 实现 Sqrt 算子，采用 shape 感知智能分核策略，根据最后一维大小动态选择按行分核或按元素分核。Kernel 层面对单轮场景使用无队列 TBuf 直接路径消除同步开销，多轮场景使用双缓冲 TQue 流水线。UB 切分严格 32B 对齐，fp16/fp32 差异化优化。
