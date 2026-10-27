# op_01_sub

提交团队: 神州数码信息系统有限公司
提交者: yuqi_711


## 算子实现介绍

本算子基于 Ascend C 在昇腾 NPU 上实现矢量减法 z = x - y，输入输出均为 (8, 2048) 的 float32/float16 张量。实现采用核函数直调模式，host 侧将 16384 个元素均分到 8 个矢量核并行处理，每核负责 2048 个元素，并按 tiling 参数在核内进一步切分为 16 块循环处理；核函数基于 TPipe/TQue 搭建双缓冲流水线，按 CopyIn（DataCopy 搬入 UB）、Compute（调用 AscendC::Sub 完成逐元素减法）、CopyOut（DataCopy 写回 GM）三段式流程执行，通过 C++ 模板按 dtype 分发到 float/half 两种实例，一份代码同时支持两种数据类型，验证集测试通过。
