# op_01_sub

提交团队: 上海昆联数码科技有限公司
提交者: luis_23333


## 算子实现介绍

基于Ascend C原生直调实现矢量减法Sub算子(z=x-y)，输入输出支持float32/float16,shape 为(8, 2048),ND 格式.实现要点:host 侧按 Tensorlnfo.dtype 分发fp32/fp16 两种核函数实例;fp16 路径先 Cast升精度到fp32计算，再以 CAST RINT舍入回写，与numpy round-half-even对齐;多核均分+核内tiling双缓冲流水，尾
块用 DataCopyPad 处理。
