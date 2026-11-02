# op_01_sub

提交团队: 杭州电子科技大学
提交者: 刘钇辰


## 算子实现介绍

Ascend C 直调实现 z = x - y。Kernel 采用 8 核并行：blockLength = totalLength / GetBlockNum() = 16384 / 8 = 2048；核内 tileNum = 8、双缓冲 BUFFER_NUM = 2，故 tileLength = 2048 / 8 / 2 = 128，单核循环 16 次正好覆盖 2048 个元素。使用 TPipe/TQue 管理 VECIN/VECOUT 队列，GlobalTensor 按 blockIdx*blockLength 做多核偏移，DataCopy 完成 GM 与 UB 间搬运，AscendC::Sub 逐元素相减，三个队列均配对 FreeTensor。模板按 dtype 选择 float / half 实例，支持 fp32 与 fp16。UB 占用约 3KB（128 x 4B x 2 buffer x 3 queue），远小于 192KB 上限。平台验证集 2/2 PASS，输出错误占比 0.00%。
