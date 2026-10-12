# challenge08_tanhcustom

提交团队: 个人开发者
提交者: skytosky


## 算子实现介绍

基于 Ascend C 实现逐元素 Tanh 激活算子：op_host 侧 Tiling 函数将输入按 8 核切分（每核均分总长度，单核内再切 16 块），支持 float32/float16 两种数据类型，通过 TilingKey 模板参数（模板组合序号 0/1）选择对应类型实例；op_kernel 侧经输入队列将数据搬入 UB，使用 Exp、Muls、Sub、Add、Div 指令计算 y=(exp(x)-exp(-x))/(exp(x)+exp(-x))，结果经输出队列搬回 Global Memory。
