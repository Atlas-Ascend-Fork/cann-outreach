# challenge08_tanhcustom

提交团队: 个人开发者
提交者: Code_demo111111


## 算子实现介绍

基于 Ascend C 单核实现。Tiling 策略上，将 totalLength 对齐到 BLOCK_SIZE 的整数倍，UB 切分按 32B 对齐以满足硬件要求。Kernel 侧采用双缓冲队列（BUFFER_NUM=2）实现流水线搬运，调用 AscendC::Tanh 接口完成逐元素计算，最终输出与输入形状完全一致的张量。
