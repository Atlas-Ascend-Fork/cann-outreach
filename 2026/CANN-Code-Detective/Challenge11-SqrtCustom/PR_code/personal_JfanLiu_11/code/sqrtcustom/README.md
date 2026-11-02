# sqrtcustom

提交者: JfanLiu

## 实现思路

- 分核方式：整网张量按核号切成 `blockFactor` 个元素的连续段，第 `i` 个核只看到 `[i * blockFactor, (i + 1) * blockFactor)`，
  尾核按剩余元素截断；`blockFactor` 先向上对齐到 32B 元素数，再粗对齐到 256 元素，避免核间任务过碎。
- 核内流水：`TQue` 输入、输出各开两块（`BUFFER_NUM = 2`），`CopyIn -> Compute -> CopyOut` 三段式由队列驱动，
  MTE2 搬运、Vector 计算与 MTE3 回写可以互相重叠。
- 尾块处理：`blockFactor` 与 `ubFactor` 都按 32B 对齐推导，仍会出现最后一块不满 `ubFactor` 的情况，
  统一用 `DataCopyPad` 承载任意字节长度，免去额外的对齐填充与手工截断。
- UB 预算：按「2 块输入 + 2 块输出」共 4 份 buffer 反推单次搬入上限，再与单核任务量取小。
- 精度：fp32 / fp16 各自直接调用 `AscendC::Sqrt`，与输入的 dtype 一致，不做额外的精度转换。
- tilingKey：`SQRT_SCH_MODE_FP16 = 0`、`SQRT_SCH_MODE_FP32 = 1`，由 op_host 按输入 dtype 选择核函数实例。

## 验证

- op_host UT：`tests/ut/op_host` 覆盖 `{8, 2048}` 的 float32 / float16 两种输入，
  校验 tilingKey、tiling 数据（16384 / 256 / 256）与 workspace 大小。
- op_kernel UT：`tests/ut/op_kernel` 以 16384 元素、单核、`ubFactor = 1024` 运行，
  覆盖 16 轮流水与尾块分支，`sqrt_data/compare_data.py` 比对 float32 输出。