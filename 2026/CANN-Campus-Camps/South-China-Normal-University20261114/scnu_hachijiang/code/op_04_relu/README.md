# op_04_relu

提交团队: 华南师范大学(石牌校区)
提交者: hachijiang


## 算子实现介绍

1. Tiling：totalNum=16384，blockFactor=CeilDiv(totalNum, 核数)（尾核取剩余），ubFactor=UB/(2 进+2 出) 后按 32B 对齐，按 dtype 选 half/float tilingKey。
2. Kernel：按 ubFactor 切块，用 DataCopyPad 搬入 → AscendC::Relu → DataCopyPad 搬出；BUFFER_NUM=2 双缓冲使搬运与计算重叠。
3. 尾块用实际 currentNum 保证不越界。
