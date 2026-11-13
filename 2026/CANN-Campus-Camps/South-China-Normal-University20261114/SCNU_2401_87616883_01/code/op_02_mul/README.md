# op_02_mul

提交团队: 华南师范大学
提交者: 蒙一豪


## 算子实现介绍

在算子工程的 Kernel 侧实现 KernelMul：Init 中按`GetBlockNum` 做多核切分、单核内按 tileNum 切块并初始化 VECIN/VECOUT 队列，Process 以 CopyIn、`AscendC::Mul` 、CopyOut 三级流水完成 z = x × y；Host 侧填充 tiling、按整除原则设置 BlockDim，并补全 Shape/Dtype 推导。
