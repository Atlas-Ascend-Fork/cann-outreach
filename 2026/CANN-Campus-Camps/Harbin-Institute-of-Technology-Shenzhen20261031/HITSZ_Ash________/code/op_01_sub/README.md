# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: 皮小雨


## 算子实现介绍

基于Ascend C实现矢量减法 z = x - y (8x2048, fp32/fp16)。

Tiling策略：采用多核+核内分块。Host侧固定启动8个核（blockDim=8），单核处理2048个元素。核内按tileNum=8, BUFFER_NUM=2分块，单次处理tileLength=128。总长度能整除，无越界。

UB对齐与搬运：单次搬运128个元素（float下为512字节，fp16下为256字节），严格满足32B对齐要求。使用TPipe与双缓冲队列（VECIN/VECOUT）实现CopyIn、Compute(Sub)、CopyOut的流水并行。
