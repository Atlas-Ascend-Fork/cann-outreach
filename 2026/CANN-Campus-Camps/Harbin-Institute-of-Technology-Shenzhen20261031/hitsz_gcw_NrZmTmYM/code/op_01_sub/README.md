# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: gcw_NrZmTmYM


## 算子实现介绍

本算子实现逐元素减法 z = x − y，支持 fp32 与 fp16，shape 展平为连续内存后处理，与维数无关。结构分三层：Host 侧的 run_kernel 解析 totalLength 生成 tiling，按 dtype 分派 half/float 模板实例，以 8 个 block 并行下发；Device 侧的 sub_custom&lt;T&gt; 构造 KernelSub&lt;T&gt;，由 Init 完成 GM 视图绑定与 UB 初始化，Process 驱动 CopyIn→Compute→CopyOut 三段流水。多核按等长切分，指针偏移前置至 SetGlobalBuffer，核内下标统一从 0 起。UB 只申请两块：x 走常规 VECIN 队列，y 由 TQueBind 绑定 VECIN/VECOUT 双身份，借助 Sub 允许目的与第二源同址的特性，计算后原地承载 z 并转输出身份入队写回，UB 占用由三块降至两块。全程单次搬运、无分块循环，同步完全交由队列机制保证。
