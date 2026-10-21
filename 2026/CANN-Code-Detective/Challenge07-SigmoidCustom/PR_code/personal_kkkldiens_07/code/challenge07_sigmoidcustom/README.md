# challenge07_sigmoidcustom

提交团队: 个人开发者
提交者: kkkldiens


## 算子实现介绍

【代码侦探Challenge07】逐元素 Sigmoid 算子：按输入类型进行32字节对齐、多核切分与UB分块，双缓冲搬运并处理尾块；float16采用float32中间计算，使用Muls、Exp、Adds和Div计算1/(1+exp(-x))。CANNJudge提交774235通过2/2测试点，输出错误占比均为0.00%。
