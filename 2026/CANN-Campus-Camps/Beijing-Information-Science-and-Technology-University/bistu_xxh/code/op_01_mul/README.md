# op_01_mul

提交团队: 北京信息科技大学
提交者: b-12a


## 算子实现介绍

本算子基于 Ascend C 编程语言实现了矢量乘法计算（Z = X * Y）。
   实现方案：
   1. 采用 TPipe 和 TQue 队列流水线编程模型，实现了 CopyIn -&gt; Compute -&gt; CopyOut 的三级流水并行。
   2. 在 Compute 阶段，调用 AscendC::Mul 核心接口完成双张量逐元素相乘。
   3. Host 侧根据总长度（totalLength）和可用核数（blockDim）自动计算 Tiling 策略，将数据按块（Tile）切分，实现了多核并行计算，最大化利用了 NPU 的算力。
