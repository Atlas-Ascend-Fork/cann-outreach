# sqrtcustom

提交者: Murasame_214

## 实现思路

- 分核方式：不做连续大段切分，而是先确定单个 tile 的长度，再把编号为 `blockIdx + k * blockDim` 的 tile 轮转分配给各核，
  核与核之间天然均衡，最后一个核不会因为尾部截断而空转。
- tile 长度两层收敛：先按 UB 容量求上限（单缓冲下每个元素同时占用输入副本、输出副本与 fp32 中间结果，
  即 `2 * typeSize + 4` 字节），再用「总元素 / 核数 / 每核 tile 数」求平衡值，两者取小，保证起核数与单次搬入量都不失控。
- 缓冲策略：`BUFFER_NUM = 1` 单缓冲，`CopyIn -> Compute -> CopyOut` 串行推进，把省下来的 UB 全部换成更长的 tile，
  减少循环轮次与同步次数。
- 搬运指令：tile 长度按 32B 对齐推导，绝大多数 tile 可以直接用 `DataCopy`；只有整网最后一个不满 tile 的尾块
  才走 `DataCopyPad`，常规路径不付 pad 的代价。
- 精度：fp16 输入先 `Cast` 升到 fp32 再 `Sqrt`，最后 `Cast` 舍入回 fp16；平方根与舍入都在 fp32 域完成，
  与 float64 参考值的偏差比半精度直算更小。fp32 输入直接单精度计算。
- tilingKey：`SQRT_CALC_MODE_FP16_UPCAST = 0`、`SQRT_CALC_MODE_FP32 = 1`，由 op_host 按输入 dtype 选择核函数实例。

## 验证

- op_host UT：`tests/ut/op_host` 覆盖 `{8, 2048}` 的 float32 / float16 两种输入，
  校验 tilingKey、tiling 数据（16384 / 256 / 128）与 workspace 大小。
- op_kernel UT：`tests/ut/op_kernel` 以 16384 元素、单核、`ubFactor = 128` 运行，
  覆盖 128 个 tile 的轮转与尾块分支，`sqrt_data/compare_data.py` 比对 float32 输出。