# op_04_relu

提交团队: 华南师范大学
提交者: 蔡仁彬


## 算子实现介绍

# Relu 算子实现介绍

## 一、算子功能

实现 `torch.nn.functional.relu` 的 Ascend C 自定义算子，对输入张量逐元素计算 `y = max(0, x)`。

- **输入 `x`**：ND 格式，shape 为 `(8, 2048)` / `(2048, 2048)`，支持 `float32` / `float16`
- **输出 `y`**：与输入同 shape、同 dtype

## 二、工程结构

```
code_temple/
├── op_host/                # 算子 Host 侧
│   ├── relu_def.cpp        # 算子原型定义（输入/输出、dtype、格式）
│   ├── relu_infershape.cpp # 形状推导（输出 shape = 输入 shape）
│   └── relu_tiling.cpp     # Tiling 逻辑（本次实现）
└── op_kernel/              # 算子 Kernel 侧
    ├── relu.cpp            # kernel 入口，按 tilingKey 实例化 float/half
    ├── relu.h              # Relu 内核类（本次实现）
    ├── relu_tiling_data.h  # Tiling 数据结构 totalNum/blockFactor/ubFactor
    └── relu_tiling_key.h   # tilingKey 定义（mode0→half, mode1→float）
```

## 三、Host 侧实现（`op_host/relu_tiling.cpp`）

### 1. 平台信息获取

通过 `GetPlatformInfo` 获取 AIV 核数 `coreNum` 与 UB 容量 `ubSize`，并初始化 workspace 大小为 0。

### 2. 总元素数与对齐单元

遍历输入 shape 求 `totalNum`；按 dtype 求 32B 数据搬移对齐单元 `alignUnit`（float32 为 8，float16 为 16）。

### 3. 启动核数配置

```cpp
blockDim = min(coreNum, totalNum / alignUnit)
context-&gt;SetBlockDim(blockDim)
```

保证每核至少处理 `alignUnit` 个元素，充分利用 910B3 的 40 个 AIV 核。

### 4. 分块参数

- `blockFactor`：每核基础处理长度，向下对齐到 `alignUnit`；无法被整除的余量统一留给 0 号核，保证所有核的搬移长度均为 32B 对齐。
- `ubFactor`：每次 UB 循环处理的元素数，按双缓冲预算计算：

  ```
  ubFactor = (ubSize - 32KB预留) / (2队列 × 2块 × sizeof(T))，向下对齐
  ```

### 5. tilingKey 选择

根据输入 dtype 选择调度模板：`float16 → mode0`（实例化 `Relu&lt;half&gt;`），`float32 → mode1`（实例化 `Relu&lt;float&gt;`）。

## 四、Kernel 侧实现（`op_kernel/relu.h`）

### 1. Init —— 多核切分与 UB 分配

- 计算每核处理的长度与 GM 起始地址（**关键的多核切分逻辑**）：
  - 0 号核：`offset=0`，处理 `blockFactor + leftover`（吸收余量）
  - 其他核：按 `blockFactor` 连续切分
- `inputGMX.SetGlobalBuffer(...)` / `outputGMY.SetGlobalBuffer(...)` 设置各核输入输出起始地址
- `pipe.InitBuffer(inputQueueX, BUFFER_NUM, ...)` 与输出队列按**双缓冲**（`BUFFER_NUM=2`）分配 UB，实现拷贝与计算的流水重叠
- 对 `ubLength_` 做 UB 预算钳制，防止 tiling 传入过大 tile 导致越界

### 2. 处理流程（Process）

```
loopCount = blockLength / ubLength;  tail = blockLength % ubLength
for i in 0..loopCount:
    CopyIn(i, ubLength) → Compute(ubLength) → CopyOut(i, ubLength)
if tail &gt; 0:
    CopyIn(loopCount, tail) → Compute(tail) → CopyOut(loopCount, tail)
```

- **CopyIn**：`DataCopy` 从 GM 搬入 UB
- **Compute**：`AscendC::Relu(yLocal, xLocal, count)` 完成 `max(0, x)`
- **CopyOut**：`DataCopy` 从 UB 搬出到 GM

## 五、关键设计与优化点

1. **多核并行**：启动全部可用 AIV 核（40 核），数据按核均分，最大化吞吐。
2. **DoubleBuffer 双缓冲**：输入/输出队列各 2 块 buffer，隐藏数据搬移与计算的等待时间。
3. **32B 对齐的尾部处理**（调试中的关键修复）：实测发现**非 32B 对齐的拷贝/计算长度会导致每核末尾元素错误**（如 `count=52` 尾部出错、`count=256` 正常）。通过将每核长度向下对齐到 `alignUnit`、余量由 0 号核吸收，保证所有 `DataCopy`/`Relu` 的 count 均为 32B 对齐，彻底解决尾部精度问题。
4. **UB 预算防御**：tiling 侧按实际 UB 容量（预留 32KB）计算 tile；kernel 侧再按 192KB UB 预算钳制，双重保护防越界。
5. **模板化调度**：按 dtype 通过 tilingKey 选择 `half`/`float` 实例，同一套代码支持两种数据类型。
