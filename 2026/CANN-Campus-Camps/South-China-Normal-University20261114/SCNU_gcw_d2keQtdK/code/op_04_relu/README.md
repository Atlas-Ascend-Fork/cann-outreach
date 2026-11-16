# op_04_relu

提交团队: 华南师范大学
提交者: gcw_d2keQtdK


## 算子实现介绍

本算子基于 Ascend C 语言开发，实现了 y=max(0,x) 的激活功能，支持 FP16 和 FP32 数据类型。实现分为 Host 侧 Tiling 与 Kernel 侧执行：
Host 侧：
获取硬件可用核数与内存上限，计算输入总元素数。
对数据进行严格的 32 字节对齐切分，计算 blockFactor（每核处理量）与 ubFactor（单次 UB 处理量），配置启动核数。
根据输入数据类型设置 TilingKey，精准路由底层模板实例。
Kernel 侧：
多核切分与安全搬运：利用 GetBlockIdx 进行多核偏移计算，并采用 DataCopyPad 接口进行数据搬运，彻底解决了尾块数据非 32 字节对齐导致的越界踩踏问题。
矢量计算：在片上缓存（UB）中调用 AscendC::Maxs 指令，高效完成 y=max(0,x) 的向量化计算。
内存管理：在队列内存分配上采用了双缓冲（Double Buffer）结构预留，保证了内存调度的灵活性。
