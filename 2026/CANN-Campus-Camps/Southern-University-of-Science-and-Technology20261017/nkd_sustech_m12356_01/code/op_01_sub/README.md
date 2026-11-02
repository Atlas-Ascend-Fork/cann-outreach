# op_01_sub

提交团队: 南方科技大学
提交者: m12356


## 算子实现介绍

用静态 Tensor 编程，不回到 TPipe/TQue；
零 tiling 入参；
不用 DataCopyPad 和尾部零头处理——16384/8=2048 正好整除、根本不存在尾块，普通 DataCopy 少了 padding 逻辑更省。
