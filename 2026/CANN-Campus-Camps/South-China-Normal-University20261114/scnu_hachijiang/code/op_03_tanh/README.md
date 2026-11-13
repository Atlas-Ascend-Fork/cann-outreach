# op_03_tanh

提交团队: 华南师范大学(石牌校区)
提交者: hachijiang


## 算子实现介绍

1. Tiling：totalNum=16384，blockFactor=CeilDiv(totalNum, 核数)（尾核取剩余），ubFactor 按 UB 容量算（2 进 + 2 出 + 3 临时共 7 块，32B 对齐），按 dtype 选 half/float tilingKey。
2. Kernel：按 ubFactor 切块，用 DataCopyPad 搬入；x 先夹到 [-8,8] 防止 exp 溢出，计算 (e^x-e^-x)/(e^x+e^-x) 后再搬出。
3. 尾核和尾块都按实际长度处理，避免越界。
