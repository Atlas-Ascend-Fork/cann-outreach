# op_04_relu

提交团队: 华南师范大学
提交者: HAJIMIO134


## 算子实现介绍

### 1. Host 侧自适应 Tiling

- 根据输入 dtype 和 shape 计算总元素数，FP16、FP32 使用不同 UB 预算。
- 地址和 tile 长度按 512 元素对齐，满足搬运带宽粒度要求。
- 小 shape 按每核 4096 元素调度；测试点 1 使用 4 个 AIV。
- 大 shape 使用双缓冲 UB 预算，并限制为 20 个 AIV，减少核调度开销。
- 测试点 2 每核约处理 `209920` 个元素，`tileLength=23040`，通过多 tile
  双缓冲持续执行。

### 2. Kernel 侧双缓冲流水

- 使用 `TQue&lt;VECIN, BUFFER_NUM&gt;` 和 `TQue&lt;VECOUT, BUFFER_NUM&gt;`，
  `BUFFER_NUM=2`。
- 多 tile 场景通过 `TQue` 的 `EnQue/DeQue/FreeTensor` 重叠相邻块的
  CopyIn、Vector Compute 和 CopyOut。
- 单 tile 场景将 `InitBuffer` 深度收缩为 1，避免申请无用的第二份 UB。

### 3. 小 shape 静态 UB 直通

- 对满足“每核恰好一个 tile”且 tile 不大于 48KB 的场景，直接使用静态
  `VECCALC` UB，不构造 `TPipe/TQue`。
- 输入和输出复用同一块 UB，显式使用 `MTE2_V`、`V_MTE3` 事件同步。
- 该路径覆盖测试点 1，减少固定框架开销和无用输出缓冲。

### 4. 对齐与尾块处理

- 全局地址和长度均按 32B 对齐时使用 `DataCopy` 快速路径。
- 非对齐时使用 `DataCopyPad`，保证尾块精度与边界安全。
- 支持空 Tensor、非整除 shape、FP16 和 FP32。
