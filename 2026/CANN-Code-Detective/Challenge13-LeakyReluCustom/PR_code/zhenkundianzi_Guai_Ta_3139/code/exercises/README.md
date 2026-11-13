# exercises

提交团队: 武汉贞坤电子有限公司
提交者: TangYunCheng


## 算子实现介绍

本算子基于 Ascend C 实现 LeakyReLU，完成逐元素计算 ，由属性 negative_slope 传入并随 tiling 数据下发至 kernel；Host 侧 tiling 根据输入元素总数、AI Core 数量、UB 容量及 dtype 计算 blockDim、blockFactor 和 ubFactor，按核切分并在核内按 UB 分块循环，同时根据 float16/float32 选择对应 tilingKey 与模板实例；Kernel 侧采用 VECIN/VECOUT 双缓冲队列，依次执行 CopyIn、Compute、CopyOut，计算时用 Maxs 取正半轴、Mins 取负半轴后再 Muls 缩放并 Add 合并，支持 ND 格式、float16 与 float32 输入输出，并对空 tensor 和尾部非整块数据做了保护处理。
