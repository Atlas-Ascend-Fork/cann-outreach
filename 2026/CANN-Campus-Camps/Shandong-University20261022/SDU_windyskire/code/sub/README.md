# sub

提交团队: 山东大学
提交者: windyskire


## 算子实现介绍

分核+UB分块双缓冲实现：totalLength按2的幂选核（≤min(availableCoreNum,32)）做等长切分；每核在UB内以128元素为tile、BUFFER_NUM=2双缓冲，TQue流水CopyIn→Sub→CopyOut；fp32/fp16模板化实例化，执行dtype分发；DataCopy按32B对齐，全程NPU向量计算，无CPU侧计算。
