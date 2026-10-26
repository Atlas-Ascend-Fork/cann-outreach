# expcustom

提交团队: 石家庄铁道大学
提交者: LRJ


## 算子实现介绍

【代码侦探Challenge10】实现 Exp 逐元素自然指数算子：支持 float16/float32，按 8 核切分 (8, 2048) 输入，使用双缓冲队列完成 CopyIn、Exp、CopyOut。统一 dtype 模板声明。CANNJudge 提交 911259 已通过验证（Pass）。
