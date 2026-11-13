# op_01_sub

提交团队: 华南师范大学
提交者: 陈俊吉


## 算子实现介绍

算子实现介绍
1.使用单核实现，UB切分按32B对齐
2.Tiling策略：totalLength对齐到BLOCK_SIZE的整数倍
3.采用模板化设计，根据输入的dtype自动适配float32/float16，复用了同一套核内逻辑
4.循环逻辑：总长度对齐后切分为8块，每次循环判断实际剩余长度，防止越界读取，并通过单缓冲串行处理（CopyIn-&gt;Compute-&gt;CopyOut）完成计算
5.计算核心：使用 AscendC::Sub 矢量减法指令实现 z = x - y 的逐元素运算
