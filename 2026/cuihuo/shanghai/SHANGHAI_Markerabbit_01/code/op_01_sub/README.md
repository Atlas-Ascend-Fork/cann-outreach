# op_01_sub

提交团队: 上海柏飞
提交者: 范国庆


## 算子实现介绍

基于Ascend C原生直调实现矢量减法Sub算子(z=x-y)，输入输出支持float32/ float16，shape为(8,2048)，ND格式。实现要点:host侧按Tensorinfo.dtype分发 fp32/fp16两种核函数实例;fp16路径先Cast升精度到fp32计算，再以CASTRINT舍入回写，与numpy round-half-even 对齐;多均分+核内tiling双冲流水，尾块用DataCopyPad处理。
