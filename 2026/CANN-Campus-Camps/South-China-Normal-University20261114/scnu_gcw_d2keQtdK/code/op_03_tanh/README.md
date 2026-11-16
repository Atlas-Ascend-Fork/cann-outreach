# op_03_tanh

提交团队: 华南师范大学
提交者: gcw_d2keQtdK


## 算子实现介绍

1. 平台信息获取与系统 Workspace 申请
通过 GetPlatformInfo 获取当前硬件（如 910B）的可用 AI Core 数量（coreNum）和片上缓存大小（ubSize）。
通过 GetWorkspaceSize 申请系统级的 Workspace（GetLibApiWorkSpaceSize()）。这是昇腾算子开发的铁律，如果不申请，底层某些 API 可能会报错（如 507035）。
2. 计算总数据量
获取输入张量的 Shape（如果是标量则转成 {1}），通过循环连乘计算出总元素个数 totalNum。同时做了 totalNum == 0 的异常保护。
3. 多核切分（Block 级）与 32 字节对齐
把总数据平均分给多个核：blockFactor = CeilDiv(totalNum, coreNum)。
关键点：昇腾 NPU 要求数据搬运（DataCopy）必须是 32 字节对齐的。所以代码计算了 alignNum = 32 / typeSize，并用 CeilAlign 把每个核要处理的数据量向上取整到 32 字节的倍数。如果取整后超了，再用 FloorAlign 退回。
4. 核内切分（UB 级）与内存防线
为了防止单次搬入的数据撑爆片上内存（UB），计算了单次最大处理量 maxUbElements。
最亮眼的设计：考虑到 float16 在计算时通常会转为 float32 进行，代码按照 ubSize / (sizeof(float) * 6) 来严格计算最大容量，并强制限制 ubFactor 不超过 4096 个元素。这从根本上杜绝了内存踩踏。
5. TilingKey 路由
根据输入的数据类型（DT_FLOAT16 或 DT_BF16 走 MODE_0，其他走 MODE_1），设置 TilingKey。底层 Kernel 会据此实例化对应的模板类。
