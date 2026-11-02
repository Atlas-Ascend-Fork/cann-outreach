# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: gcw_H1nrGbmJ


## 算子实现介绍

本算子基于 Ascend C Direct Invocation 模式实现 `z = x - y` 矢量减法：host 侧 `run_kernel` 从 `TensorInfo` 累乘形状得到 `totalLength`（16384）并设 `tileNum=8` 组成 `SubCustomTilingData` 传入核函数，核内 `KernelSub&lt;T&gt;` 在 `Init` 中按 `GetBlockNum()` 将数据均分为 8 核（每核 `blockLength=2048`），再用 `GetBlockIdx()` 偏移 `SetGlobalBuffer` 完成核间切分，并按 `tileLength = blockLength/tileNum/BUFFER_NUM = 128` 为三个队列分配 Double Buffer 的 UB 内存；`Process` 循环 `tileNum×BUFFER_NUM=16` 次，依次执行 CopyIn（`DataCopy` GM→UB 并 `EnQue`）、Compute（`DeQue` 后调用 **`AscendC::Sub`** 做逐元素减法再 `EnQue` 并释放输入）、CopyOut（`DeQue` 后 `DataCopy` UB→GM 并释放），形成搬入/计算/搬出三级流水且 UB 无泄漏；整体框架完全复用 Add 参考代码，仅将结构体、类名、核函数名及计算原语 `Add` 替换为 `Sub`，并通过模板化 `&lt;T&gt;` 与 `sizeof(T)` 自动适配 fp16/fp32，保证 8 核并行覆盖全部 16384 元素、地址无重叠无越界。
