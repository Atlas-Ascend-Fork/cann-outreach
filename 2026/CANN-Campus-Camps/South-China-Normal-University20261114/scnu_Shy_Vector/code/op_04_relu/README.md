# op_04_relu

提交团队: 华南师范大学
提交者: Shy_Vector


## 算子实现介绍

## 1. 功能与整体框架

本算子实现 `y = max(0, x)`，按昇腾 CANN 自定义算子规范分为 Host 与 Kernel 两层：

```
op_host/relu_tiling.cpp     → Host 侧 Tiling：计算核数、每核数据量、UB 循环粒度
op_kernel/relu.h            → NPU 侧 Kernel：GM↔UB 搬运 + 矢量计算，双缓冲流水
op_kernel/relu.cpp          → Kernel 入口，按 tilingKey 分发 fp16 / fp32 模板
op_kernel/relu_tiling_key.h → 模板参数定义（fp16 / fp32 两个编译模式）
```

支持 `(8, 2048)` / `(2048, 2048)` 的 float32 / float16，数据格式为 ND。

核心接口：

| 参数 | 类型 | 形状 | 数据类型 | 格式 |
|---|---|---|---|---|
| 输入 x | tensor | (8,2048) / (2048,2048) | float32 / float16 | ND |
| 输出 y | tensor | (8,2048) / (2048,2048) | float32 / float16 | ND |

## 2. Tiling 策略（op_host/relu_tiling.cpp）

按运行时平台信息（AIV 核数、UB 大小）和输入形状动态切分：

| 参数 | 含义 | 计算方式 |
|---|---|---|
| `totalNum` | 总元素数 | 各维连乘 |
| `blockDim` | 启动核数 | 数据量 ≤128KB 用**单核**；否则用满全部 AIV 核 |
| `blockFactor` | 每核元素数 | `ceil(totalNum / blockDim)` 再按 32B 对齐 |
| `ubFactor` | 每次 UB 循环元素数 | 按 UB 预算（90%）÷ 4 份 buffer ÷ 类型字节，32B 对齐 |

**关键优化**：`(8, 2048)` fp32 仅 64KB，数据量小于单核吞吐阈值时使用 1 个核直接处理，省掉多核块调度开销（实测该场景耗时下降约 15%）；`(2048, 2048)` 则用满 40 核并行，每核约 400KB。

## 3. Kernel 流水（op_kernel/relu.h）

### 3.1 双缓冲流水

输入/输出队列各 `BUFFER_NUM = 2` 份 buffer，`CopyIn → Compute → CopyOut` 三段通过队列事件同步自动重叠：

- 第 i+1 块的 `DataCopy`（MTE2 搬运）与第 i 块的矢量计算（V）、写出（MTE3）并行；
- 搬运、计算、写出三条流水同时工作，充分掩盖访存延迟。

### 3.2 主循环快速通路

把循环拆成**整块快速通路** + **尾块处理**：

```cpp
fullLoops = coreDataNum_ / ubLength_;
for (i &lt; fullLoops) { CopyInAll(progress); Compute(ubLength_); CopyOutAll(progress); }
if (tailNum &gt; 0)    { CopyIn(progress, tailNum); Compute(tailNum); CopyOut(progress, tailNum); }
```

- 整块通路直接用整块 `DataCopy`，去掉每轮迭代的除法和尾块分支判断，减少标量流水开销；
- 由于 `ubFactor` / `blockFactor` 均按 32B 对齐，本算例各块完全对齐，走纯 `DataCopy` 最高效通路；
- 尾块走 `DataCopyPad` 精确搬运，保留对非对齐 shape 的通用正确性。

### 3.3 核间切分

每个核 `blockIdx` 从 `blockIdx * blockFactor` 处开始，处理连续的一段元素；每核重复“整块循环 + 尾块”，各核间数据互不重叠。

## 4. 优化验证结果（本机 40 AIV / UB 192KB）

| 指标 | 优化前 | 优化后 |
|---|---|---|
| (8,2048) fp32 单次调用 | 8.22 μs | 6.99 μs |
| (2048,2048) fp32 设备核时 | 9.38 μs | 9.28 μs |

- 大数据场景已接近流水瓶颈：实测 MTE2 搬运占用约 82%，对 buffer 数、不对称缓冲、核数、UB 占比多组实验均无进一步收益，当前结构即局部最优。
- 正确性验证：设备样例 PASS（0/16384 错）、Host Tiling UT PASS、Kernel CPU 仿真 UT PASS。
