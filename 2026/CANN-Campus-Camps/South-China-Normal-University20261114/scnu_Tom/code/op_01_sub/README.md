# op_01_sub

提交团队: 华南师范大学
提交者: ljt


## 算子实现介绍

本算子基于Ascend C编程语言原生开发，实现矢量减法功能（z = x - y），在昇腾NPU上高效运行。具体实现策略如下：
单核调度实现：根据赛题要求，采用单核（Single Core）模式启动Kernel（blockDim设为1），由单一的AI Core完成全量数据的搬运与计算，降低了多核切分的复杂度，保证了计算逻辑的稳定性。
UB切分与32B对齐：在Tiling策略中，严格按照硬件UB（Unified Buffer）的访问要求进行32字节（BLOCK_SIZE）对齐处理。根据输入数据类型的不同（float32或float16），动态计算每次搬运的对齐元素个数（32B / sizeof(T)），并保证Local Tensor的切分长度严格为32B的整数倍，避免非对齐访问带来的性能损失或内存越界风险。

Tiling策略：总数据量 totalLength（8*2048=16384）向上对齐到 BLOCK_SIZE 的整数倍。由于数据量较小，将总长度均匀切分为 tileNum 个分块进行循环处理，配合双缓冲（Double Buffering）机制（BUFFER_NUM=2），流水化执行 CopyIn、Compute（AscendC::Sub）、CopyOut，实现计算与搬运的并行，保证算子整体性能。
多类型支持：采用模板类（Template）设计，通过 
run_kernel
 阶段解析 TensorGroupInfo 中的 dtype，动态派发至 float32 或 float16 的Kernel实例，确保数据类型计算精度及接口要求完全匹配。
