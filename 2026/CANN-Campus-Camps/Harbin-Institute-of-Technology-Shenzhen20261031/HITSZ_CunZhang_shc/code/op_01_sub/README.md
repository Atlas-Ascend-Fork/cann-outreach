# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: CunZhang_shc_


## 算子实现介绍

基于 AscendC 实现 z = x - y。Host 侧由输入 shape (8, 2048) 累乘得到 totalLength = 16384，按 dtype 选择 float、half 模板实例，blockDim = 8 启动。Kernel 侧按 GetBlockNum() 将数据均分，每核 blockLength = 2048 元素；再按 tileNum = 8、BUFFER_NUM = 2 切分为每块 128 元素，fp32 下为 512B、fp16 下为 256B，均满足 32B 对齐，totalLength 为 BLOCK_SIZE 的整数倍，不存在尾块越界。每块数据按 CopyIn → Compute → CopyOut 三级流水处理，Compute 调用 AscendC::Sub(z, x, y) 完成逐元素相减；输入输出队列各分配 BUFFER_NUM 块 UB 空间实现双缓冲，计算后及时 FreeTensor 释放，避免 UB 泄漏。
