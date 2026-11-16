# op_01_sub

提交团队: 华南师范大学
提交者: 刘佳琪


## 算子实现介绍

本算子实现了基础的矢量减法（z = x - y）。
1. Tiling与多核策略：根据输入张量(8, 2048)共16384个元素，启动8个AI Core并行计算，每个核分配2048个元素，实现计算负载均衡。
2. 流水线设计：采用双缓冲（BUFFER_NUM=2）机制，利用VECIN和VECOUT队列，实现CopyIn、Compute、CopyOut的三级流水线并行，有效掩盖全局内存访问延迟。
3. 计算逻辑：在Compute阶段调用AscendC内置的Sub API完成Local Tensor的逐元素相减。
4. 数据类型支持：支持fp32和fp16数据类型，根据Host侧传入的dtype动态分发对应的模板实例。
