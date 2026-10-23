# atanh — AscendC 算子

基于 **Ascend C** 框架实现的 `atanh`（反双曲正切）向量算子。

## 算子规格 (Spec)

| 项 | 说明 |
| --- | --- |
| 输入 | `x` — 1 个 |
| 输出 | `y` — 1 个 |
| Shape | 4 维 `[N4, N3, N2, N1]` |
| 数据类型 | `float16` (half) |
| 数据格式 | `ND`（`N1` 为内维 / 连续维） |
| 数学定义 | `y = atanh(x) = 0.5 * ln((1 + x) / (1 - x))` |

## 设计说明

1. **无原生指令**：Ascend C 基础矢量数学集中没有 `atanh` 单指令，因此用
   `atanh(x) = 0.5 · ln((1+x)/(1-x))` 这一恒等式，由受支持的基础指令
   `Cast / Duplicate / Add / Sub / Div / Ln / Muls` 组合实现。

2. **精度**：超越函数在 **float32** 下计算，结果再转回 **float16**。
   直接在 float16 里算会让中间量 `(1+x)/(1-x)` 在 `|x|` 接近 1 时上溢，
   且中段精度也不足。fp32 计算 + fp16 输出与真实 atanh 核函数一致，
   误差仅为 float16 舍入。

3. **布局**：ND 布局且内维 `N1` 连续，整个 4 维张量等价于一段
   `totalLength` 个 half 的平坦流。核函数把 `x / y` 当作 1-D 处理，
   在 AI Core 之间切分，末尾（`coreLength`）由最后一个 core 处理。

4. **数据流（矢量编程范式，三级流水）**，每个 tile：

   ```
   CopyIn   : DMA  x tile   Global -> UB (VECIN 队列)
   Compute  : 矢量运算      UB     -> UB (VECCALC 队列)
   CopyOut  : DMA  y tile   UB     -> Global (VECOUT 队列)
   ```

   跨 core 并行由 `GetBlockIdx()/GetBlockNum()` 完成，
   跨 DMA / 矢量单元的同步由队列的 `EnQue/DeQue` 完成。

## 目录结构

```
atanh_op/
├── README.md
├── CMakeLists.txt                 # Ascend C 核函数工程
├── op_kernel/
│   ├── atanh_tiling.h             # 共享 tiling 结构体 (host / kernel 共用)
│   └── atanh.cpp                  # Ascend C 核函数（核心实现）
├── op_host/
│   └── atanh_tiling.cpp           # host 侧 tiling 计算
└── python/
    └── atanh_cpu.py               # CPU 参考实现 + 测试数据生成
```

## 构建

依赖已安装 CANN（Ascend Toolkit / CANN 社区版）并配置好环境变量
（`ASCEND_HOME`、`CANN_CVK` 等）。核函数使用 Bisheng / tikcpp 编译：

```bash
cd atanh_op
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
# 产物：libatanh_kernel.so
```

> 不同 CANN 版本的 cmake 模块路径略有差异；如 `find_package(AscendC)`
> 在你的版本中不可用，直接参考 `CMakeLists.txt` 里的编译命令即可手动编译
> `op_kernel/atanh.cpp`。

## 测试 / 验证

1. 生成输入与期望输出（float16，4 维 ND）：

   ```bash
   cd atanh_op/python
   python atanh_cpu.py gen --shape 2,4,8,16 --out-dir ../data
   ```

2. 在 NPU 上跑核函数（通过你的算子调用 / 图引擎 / kernel 直调）得到 `output.npy`。

3. 比对核函数输出与 CPU 参考：

   ```bash
   python atanh_cpu.py check --input ../data/input.npy --output ../data/output.npy
   # -> result = PASS
   ```

参考实现与核函数使用同一恒等式（fp32 计算、fp16 输出），故正确的核函数输出
应在 float16 舍入误差内一致（默认容差 `rtol=atol=1e-2`）。

## 接入真实 CANN 算子框架

`op_host/atanh_tiling.cpp` 中的 `AtanhTiling()` 是版本无关的 tiling 核心逻辑。
把它接到具体 CANN 版本的算子定义（`op_proto/*.proto`、TBE / 自定义算子
host 入口、算子注册与 python 绑定）即可成为完整的 CANN 自定义算子。
文件末尾附有典型 tiling 入口的示例签名。
