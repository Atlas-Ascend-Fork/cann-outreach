# add_custom 算子（Ascend C Kernel 直调）

`z = x + y` 的 Ascend C 逐元素加法算子，同时提供 **Kernel 直调可执行程序** 与 **PyTorch 算子接入** 两条通路。

| 项目 | 规格 |
|---|---|
| 算子名称 | `add_custom` |
| 计算 | `z = x + y`（逐元素） |
| 输入 | `x`、`y`：2 维张量 `[N2, N1]` |
| 输出 | `z`：与输入同 shape / 同 dtype |
| 数据类型 | float16 |
| 数据格式 | ND |
| 编程框架 | Ascend C |
| 目标架构 | Atlas A2（`dav-2201`，可在 `CMakeLists.txt` 改为 `dav-3510`） |

## 实现要点

- **三段式数据流**：`CopyIn(MTE2) → Compute(Vector) → CopyOut(MTE3)`，三条队列各双缓冲，搬运与计算流水并行。
- **Tiling 切分**：`ComputeAddCustomTiling()` 按实际 Vector 核数把总元素数切成 `blockNum` 段，前 `blockNum-1` 个核各处理 `TILE_LENGTH` 整数倍的元素，最后一个核处理余量（尾块）。该函数写在 `op_kernel/add_custom_tiling.h` 中，host 直调与 PyTorch 接入共用，避免两份 tiling 逻辑漂移。
- **2 维形状处理**：ND 格式下 `[N2, N1]` 在 GM 中按行优先连续排布，逐元素加法与形状无关，核内按连续一维区间切分；`N2/N1` 保存在 tiling 结构体中，并由 host 侧做形状校验，后续如需改为按行切分可直接使用。
- **尾块**：单 tile 元素数固定为 `TILE_LENGTH`，最后一个 tile 可能不足；用 `DataCopyPad` 搬运，保证非 32B 对齐的尾块也能正确读写。
- **UB 占用**：`TILE_LENGTH = 8192`（float16 → 16 KB/块），3 队列 × 双缓冲 = 96 KB，低于 Atlas A2 的 192 KB。

## 文件结构

```
add_custom/
├── op_kernel/
│   ├── add_custom_tiling.h      Tiling 常量 + 结构体 + tiling 计算（host/kernel 共用，纯 C++）
│   └── add_custom_kernel.asc    Kernel 实现（KernelAddCustom 类 + add_custom_kernel 核函数入口）
├── op_host/
│   ├── add_custom.asc           Host 直调入口 + main（含用例 shape [512, 1024]）
│   └── data_utils.h             .bin 文件读写工具
├── op_extension/
│   ├── add_custom_torch.cpp     PyTorch host 实现（规格校验 + tiling + kernel launch）
│   ├── register.cpp             TORCH_LIBRARY 注册（PrivateUse1 + Meta backend）
│   └── ops.h                    函数声明
├── scripts/
│   ├── case_config.py           直调用例 shape（DIM0/DIM1，须与 op_host/add_custom.asc 一致）
│   ├── golden.py                golden 计算（float32 参考，两条通路共用）
│   ├── gen_data.py              生成 float16 输入与 golden
│   ├── verify_result.py         直调通路精度验证
│   └── test_torch.py            PyTorch 通路多用例测试
├── CMakeLists.txt               双 target：可执行文件 + libadd_custom_ops.so
├── run.sh                       一键流程
└── README.md
```

## 快速开始

前置条件：已安装 CANN Toolkit 与 `torch`、`torch_npu`，且 `ASCEND_HOME_PATH` 已配置。

### 方式一：直调验证（可执行文件）

```bash
bash run.sh
```

等价的手动步骤：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build && cd build && cmake .. && make -j
python3 ../scripts/gen_data.py
./add_custom
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

运行时会打印实际切分结果，例如：

```
[info] shape=[512, 1024], dtype=float16, total=524288, coreNum=48, blockNum=..., numPerCore=..., tailNumLastCore=...
```

### 方式二：PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libadd_custom_ops.so")

x = torch.randn(512, 1024, dtype=torch.float16).npu()
y = torch.randn(512, 1024, dtype=torch.float16).npu()
z = torch.ops.npu.add_custom(x, y)

expected = x.float() + y.float()
assert torch.allclose(z.cpu().float(), expected, rtol=1e-3, atol=1e-3)
```

仅跑 PyTorch 通路：`bash run.sh --torch`。

## 精度判据

参考结果统一用 **float32** 计算（两个 float16 之和在 float32 中精确可表示），校验时把 float16 输出升到 float32 后比较：

```
|got - golden| <= atol + rtol * |golden|,  rtol = atol = 1e-3
```

float16 机器精度 `eps = 2^-10 ≈ 9.77e-4`，因此该容差等价于「允许一次 float16 舍入」，既能容纳正常的舍入误差，又不会掩盖真实的计算错误（例如尾块越界、核间区间错位）。

## 测试覆盖

| 用例 | shape | 覆盖点 |
|---|---|---|
| T1 random 2D main | `[512, 1024]` | 主用例，多核 + 整 tile |
| T2 unaligned tail | `[127, 33]` | 尾块非 32B 对齐 |
| T3 tiny 8x16 | `[8, 16]` | 单核单 tile |
| T4 single row | `[1, 8192]` | 单行 |
| T5 single elem | `[1, 1]` | 最小输入 |
| T6 zeros | `[512, 1024]` | 全零 |
| T7 pos_neg | `[256, 512]` | 正负抵消，结果应精确为 0 |
| T8 large | `[2048, 2048]` | 4M 元素，多核多 tile |
| T9 large values | `[64, 64]`，幅值 ×100 | 大数量级下的 float16 舍入 |

## 常见改动点

- **换 shape**：改 `op_host/add_custom.asc` 的 `kDim0/kDim1`，并同步 `scripts/case_config.py` 的 `DIM0/DIM1`；PyTorch 通路无需改动（按张量实际形状计算）。
- **换数据类型**：改 `op_kernel/add_custom_kernel.asc` 中的 `half`、`add_custom_tiling.h` 的 `DTYPE_BYTES`、`op_extension/add_custom_torch.cpp` 的 dtype 校验、`scripts/*.py` 的 dtype 与容差。
- **调性能**：调大/调小 `TILE_LENGTH`（需保证单核 UB 各队列总量不超限），或把按一维连续区间切分改为按行切分（`tiling.dim0/dim1` 已下发）。
- **换架构**：`CMakeLists.txt` 中两处 `--npu-arch=dav-2201` 改为目标架构（如 `dav-3510`）。
