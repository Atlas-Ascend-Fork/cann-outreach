# expcustom

提交团队: 个人开发者
提交者: HansonZhang


## 算子实现介绍

Exp 逐元素自然指数算子开发：`y = e^x`，输入 `x` 与输出 `y` 同 dtype（float32 / float16）、同 shape（ND 格式）。

### 实现要点

| 文件 | 说明 |
|:--|:--|
| `op_host/exp_def.cpp` | 算子定义：单输入 `x`、单输出 `y`，均支持 float32/float16，ND 格式（模板已完整，未改动） |
| `op_host/exp_infershape.cpp` | 形状推导：一元算子，输出 shape = 输入 shape |
| `op_host/exp_tiling.cpp` | TilingFunc：计算 `totalNum` / `blockFactor` / `ubFactor`，按 32 字节对齐后 `SetBlockDim`，并按 dtype 选择 tilingKey（0=float16，1=float32）；workspace 占用为 0 |
| `op_kernel/exp_tiling_data.h` | `ExpTilingData { totalNum, blockFactor, ubFactor }`，字段名与官方单测 `test_exp.cpp` 保持一致 |
| `op_kernel/exp_tiling_key.h` | tiling 模板参数 `schMode`（0=float16，1=float32），与官方单测中的 `exp<1>` / `expectTilingKey == 1UL` 对齐 |
| `op_kernel/exp.cpp` | Kernel 入口，按 `schMode` 分发到 `NsExp::Exp<half>` / `NsExp::Exp<float>` 两个实例 |
| `op_kernel/exp.h` | 算子核心类：`Init` / `CopyIn` / `Compute` / `CopyOut` / `Process` 的完整实现 |

### 关键设计

1. **float32 路径**：直接调用 Ascend C 向量 API `AscendC::Exp(dst, src, count)`。
2. **float16 路径**：`AscendC::Exp` 按 float32 精度计算，故先 `Cast` 到 fp32 求 Exp，再
   `Cast` 回 fp16（`CAST_NONE` 提升精度、`CAST_RINT` 舍入回写），
   中间 fp32 缓冲 `floatBuf` 仅在 fp16 实例中申请 —— host 侧的 UB 预算已把这 4 字节/元素算进去。
3. **32 字节对齐（两条路径）**：host 侧把 `blockFactor` / `ubFactor` 按 `CeilAlign(..., 32 / typeSize)`
   对齐；kernel 侧的**末尾 tile 可能不满一个对齐单元**（如 `totalNum=1001` 的 float32 尾核只有 9 个元素），
   这类块改走 `DataCopyPad`（DMA 长度按字节给出），UB 容量则一律按对齐后的 `ubFactor` 申请。
4. **越界保护**：每个核只处理 `blockFactor` 个元素，末尾核按 `totalNum - offset` 截断；
   超出数据范围的核 `blockLength_ = 0`，不进入计算循环。
5. **tile 划分**：模板 `Process()` 骨架保持不变（循环次数 = `tileNum × BUFFER_NUM`），
   本实现取 `BUFFER_NUM = 1`，`tileNum = ceil(blockLength / tileLength)`，
   因此一个核可以有多个 tile（当 `blockFactor > ubFactor` 时）。

### 已知限制与未验证项（如实说明）

- **无实机/编译器证据**：本提交在无 NPU、无 CANN toolkit 的环境中完成，
  `build.sh`、`tests/ut/run.sh`、`examples/test_aclnn_exp.cpp` **均未执行过**。
  已完成的检查只有：改动范围审计（只有已知的 TODO 文件不同）、
  官方判定脚本逐字节一致、以及用 numpy 复刻 host tiling + kernel 分块逻辑的数值仿真
  （覆盖性/对齐性/UB 占用，含反向对照）。
- **官方 host UT 的 `expectTilingData = "0 1 0 "` 是脚手架占位值**
  （恰等于 `ExpTilingData` 三个字段的默认值 `0 / 1 / 0`）。
  按模板自己的注释「修改 tiling 逻辑后请更新」，真实实现会输出 `16384 256 256`，
  因此 `tests/ut/run.sh` 中该用例**必然报 1 项失败**。
  这里**故意不修改官方测试**（改判定脚本 = 自改考卷），
  有效判据应看 kernel 侧 CPU 仿真与精度比对。
- 官方 kernel UT 只覆盖「单核 + 单 tile + 输入恒为 1.0 + float32」，
  **多核切分、尾块、float16 分支未被该 UT 覆盖**。


### 未修改的文件

`examples/`、`tests/ut/`、`CMakeLists.txt`、`build.sh` 等官方验证脚本与构建脚本均保持与模板逐字节一致，
未做任何改动。
