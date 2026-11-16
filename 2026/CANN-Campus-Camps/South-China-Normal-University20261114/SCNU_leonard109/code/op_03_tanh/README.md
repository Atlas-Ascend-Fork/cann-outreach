# op_03_tanh

提交团队: 华南师范大学
提交者: leonard109


## 算子实现介绍

op_host/tanh_tiling.cpp
    根据实际输入长度、AIV 核数和 UB 容量计算 totalNum/blockFactor/ubFactor。
    核间边界以 16 个元素对齐；每核尽量至少处理 1024 个元素。
    每次 UB 计算最多 2048 个元素，并核算双缓冲队列和两个 float32 临时张量。
    float16 选择 key 0，float32 选择 key 1；不再将 BF16 错当成 float16。
  op_kernel/tanh.h
    完成 Init、CopyIn、Compute、CopyOut、Process。
    使用双缓冲队列及显式矢量流水同步。
    普通块使用 DataCopy，非 32 字节对齐尾块使用 DataCopyPad。
    尾部仅写有效元素，空张量不进行数据搬运和计算。
