# op_01_sub

提交团队: 华南师范大学
提交者: gcw_oCPnZPED


## 算子实现介绍

实现思路：采用多核并行 + 核内分块（Tiling）的矢量减法算子。
1. Tiling 策略：由 host 侧 run_kernel 遍历输入 TensorInfo 各维求出总元素个数 totalLength（8×2048=16384），单核分块数 tileNum=8，以 tiling 结构体按值传入核函数。
2. 多核切分：启动 blockDim=8 个核，每核处理 blockLength = totalLength / 8 = 2048 个元素，各核通过 GetBlockIdx() 偏移互不重叠地访问 Global Memory。
3. 核内切分：单核内再切 tileNum×BUFFER_NUM=16 块，每块 tileLength = 2048/16 = 128 个元素，满足 32B 对齐要求（fp16 为 256B、fp32 为 512B）。
4. 数据流：CopyIn 用 DataCopy 将 x、y 各一块从 GM 搬入 UB → Compute 调用矢量减法指令 Sub(zLocal, xLocal, yLocal, tileLength) 完成 z = x - y → CopyOut 将结果搬回 GM；输入输出队列采用 Double Buffer（BUFFER_NUM=2）提升搬运与计算的并行度。
5. 数据类型：host 侧按输入 dtype 分派，fp16 使用 half 模板实例，fp32 使用 float 模板实例。
评测结果：2/2 测试点通过，输出错误占比 0%。
