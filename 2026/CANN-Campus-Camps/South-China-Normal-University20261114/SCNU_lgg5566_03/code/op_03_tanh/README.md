# op_03_tanh

提交团队: 华南师范大学
提交者: HAJIMIO134


## 算子实现介绍

## 二、算子实现

- 计算公式：tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
- fp16 输入统一 Cast 到 fp32 计算，再一次 Cast 回 fp16，避免半精度 Exp 溢出
- 大值先 Mins 到 20，防止 exp(2x) 溢出
- 多核切分：8 个 AIV 核 × 2048 元素（16384 恰好整除），按 64 元素对齐
- Kernel：每核 1 块、BUFFER_NUM=1；计算在输出 buffer 上原地完成，仅占 1 块临时 buffer
- Host Tiling：totalNum / blockFactor / ubFactor；schMode 0=fp16、1=fp32
- workspace = 0

## 三、性能优化（Ascend 910B3，CANN 9.0.0）

按第 8 章性能优化方法，用 msprof 采集数据、定位瓶颈后逐项优化：

- 核数扫描（均摊）：40→4.56 µs、20→2.69、16→2.49、12→2.22、10→2.10、
  9→2.07、8→2.01 µs，最终选 8 核
- 每核分块：1 块最优（2 块 2.91、4 块 3.36、8 块 4.45 µs）
- 小算子调用：19 条（多项式 + exp + Select）→ 9 条（exp + 近零保护）→
  6 条（exp 纯公式），Vector 耗时 0.69→0.276 µs
- 其他验证：512B 对齐、合并 TBuf、手写 SetFlag/WaitFlag、AIV_ONLY 均无额外收益
- 最终均摊耗时：1.99 µs(fp32) / 2.10 µs(fp16)
- 单次 TaskDuration：2.56–2.80 µs
- 同结构纯拷贝底线 1.87 µs，计算部分仅剩 0.12–0.23 µs
