# op_01_sub

提交团队: 上海昇腾生态创新中心
提交者: TensorLake


## 算子实现介绍

基于 Ascend C 原生直调实现矢量减法 Sub 算子（z = x - y），输入输出支持 float32 / float16，shape 为 (8, 2048)，ND 格式。实现要点：host 侧按 TensorInfo.dtype 分发 fp32/fp16 两种核函数实例；fp16 路径先 Cast 升精度到 fp32 计算，再以 CAST_RINT 舍入回写，与 numpy round-half-even 对齐；多核均分 + 核内 tiling 双缓冲流水，尾块用 DataCopyPad 处理。
