# expcustom

提交团队: 个人开发者
提交者: weixin_48190404


## 算子实现介绍

基于Ascend C实现Exp自然指数算子。Host侧完成Tiling切分，按硬件核心数均分数据并做32B对齐，计算得出blockFactor与ubFactor；Kernel侧使用while循环处理尾块，完成GM到UB的数据搬运，调用AscendC::Exp指令进行逐元素指数计算，并将结果写回GM。支持float32和float16数据类型。
