# op_02_mul

提交团队: 华南师范大学
提交者: gcw_d2keQtdK


## 算子实现介绍

1. Host 侧 (CPU端)：规划与配置
Shape/Type 推导：根据输入 x 的形状和数据类型，自动推导输出 z 的形状（与 x 一致）和类型。
Tiling 切分：获取总数据长度，将其切分成 8 个分块（tileNum），并指定分配 8 个 NPU 核（blockDim）去并行计算。
2. Kernel 侧 (NPU端)：流水线执行
多核切分：每个核根据自身的 blockIdx 认领总数据中属于自己的那一段。
双缓冲流水线 (Double Buffer)：利用 TQue 队列和 TPipe 内存管理，将数据块按流水线节奏循环处理：
CopyIn：把数据从全局内存（GM）搬运到片上高速缓存（UB）。
Compute：调用矢量计算指令 AscendC::Mul，在 UB 中完成两个数据块的乘法。
CopyOut：把算好的结果搬回全局内存。
