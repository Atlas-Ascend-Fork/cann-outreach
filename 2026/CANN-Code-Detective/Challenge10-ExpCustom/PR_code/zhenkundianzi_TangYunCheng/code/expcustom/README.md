# expcustom

提交团队: 武汉贞坤电子有限公司
提交者: TangYunCheng


## 算子实现介绍

Exp 算子实现 torch.exp 的逐元素自然指数计算：
Host 侧在 exp_tiling.cpp 按输入元素数做多核均分和 32B 对齐切分（blockFactor 为每核元素数，ubFactor 为 UB 单次循环元素数，按 UB 容量与双缓冲预算推算），并根据 dtype 设置 TilingKey（fp16 走 schMode 0，fp32 走 schMode 1）；Kernel 侧 exp.h 按 blockIdx 切出本核负责的 GM 区间，用 TQue 双缓冲流水循环执行 CopyIn→Compute→CopyOut，核心计算调用 AscendC 内置 Exp 向量指令，尾块用 currentNum 精确控制避免越界，空 tensor 直接返回。
