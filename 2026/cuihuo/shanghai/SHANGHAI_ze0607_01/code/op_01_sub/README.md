# op_01_sub

提交团队: 珑京信息科技有限公司
提交者: ze0607


## 算子实现介绍

基于AscendC原生直调实现矢量减法Sub算子(z=x-y)，输入输出支持float32/float16,shape 为(8, 2048),ND格式.实现要点:host 侧按 Tensorlnfo.dtype 分发fp32/fp16两种核函数实例;fp16 路径先Cast升精度到fp32计算，再以CAST_RINT舍入回,与numpy round-half-even 对齐;多核均分+核内tiling双缓冲流水,尾块用 DataCopyPad 处理。
