# add_custom 自定义算子（Ascend C）

基于 **Ascend C** 编程语言的逐元素加法自定义算子。

## 1. 算子规格

| 项目 | 说明 |
| --- | --- |
| 算子名称 / OpType | `add_custom` / `AddCustom` |
| 功能 | 对两个输入张量 `x`、`y` 执行逐元素加法，返回 `z` |
| 数学公式 | `z = x + y` |
| 输入 | `x`、`y`（均为 float16，ND） |
| 输出 | `z`（float16，ND） |
| shape | 2 维，如 `[N2, N1]`（元素总数 `N2 * N1`） |
| 数据类型 | float16（`DT_FLOAT16` / 设备侧 `half`） |
| 数据格式 | ND |
| 核函数类型 | AIV（纯矢量计算） |

## 2. 目录结构

```
add_custom/
├── op_kernel/
│   ├── add_custom.cpp          # 核函数（Ascend C 矢量算子实现）
│   ├── add_custom_tiling.h     # TilingData 结构体（Host 与核函数共用）
│   └── CMakeLists.txt          # 核函数构建脚本（参考）
├── op_host/
│   ├── add_custom.cpp          # TilingFunc + InferShape/InferDataType + OpDef 注册
│   └── CMakeLists.txt          # Host 侧构建脚本（参考）
├── build.sh                    # 构建入口（参考）
└── README.md
```

## 3. 实现说明

### 3.1 核函数 `op_kernel/add_custom.cpp`
采用 Ascend C 的**矢量三级流水 + 双缓冲**编程范式，`Process()` 中循环执行：

1. `CopyIn`：`DataCopy` 把第 `i` 个块从 Global Memory 搬到 Local Memory，并 `EnQue` 入输入队列；
2. `Compute`：`DeQue` 取出 `x`、`y`，调用矢量双目 `Add` 计算 `z = x + y`，结果 `EnQue` 入输出队列，并释放输入 LocalTensor；
3. `CopyOut`：`DeQue` 取出 `z`，`DataCopy` 写回 Global Memory，并释放 LocalTensor。

数据通过 `TQue<QuePosition::VECIN/VECOUT, BUFFER_NUM>`（`BUFFER_NUM = 2`）在搬运与计算之间重叠，实现流水线并行。

### 3.2 数据并行（Tiling）切分
两级切分，切分参数由 Host 侧 `TilingFunc` 计算并写入 `AddCustomTilingData`：

- **核间（Block）**：整体数据按 `NUM_BLOCKS` 个 AICore 均分，单核起始地址偏移 `blockLength * GetBlockIdx()`；
- **核内（Tile）**：单核数据按 `tileNum * BUFFER_NUM` 个块循环处理。

以 `[8, 2048]`（16384 元素）为例：`blockLength = 16384 / 8 = 2048`，`tileNum = 8`，
`tileLength = 2048 / 8 / 2 = 128`，单核循环 `8 * 2 = 16` 次，每次处理 128 个元素。

> 核函数入口参数顺序固定为「输入、输出、workspace、tiling」：
> `add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)`。
> Host 侧通过 `context->GetTilingData<AddCustomTilingData>()` 写字段，核函数侧通过
> `REGISTER_TILING_DEFAULT(AddCustomTilingData)` + `GET_TILING_DATA(tilingData, tiling)` 读取。

### 3.3 Host 侧 `op_host/add_custom.cpp`
- `TilingFunc`：读取输入 `x` 的总元素数，设置 `SetSimdNumBlocks(NUM_BLOCKS)`，写入 `totalLength` / `tileNum`，workspace 置 0；
- `InferShape`：输出 `z` 的 shape 跟随输入 `x`（逐元素运算不改变形状）；
- `InferDataType`：输出 `z` 的 dtype 跟随输入 `x`；
- `class AddCustom : public OpDef`：声明输入 `x`/`y`、输出 `z`（均为 float16 / ND），关联 Tiling 与推导函数，并 `OP_ADD(AddCustom)` 注册。

## 4. 构建与验证

> **推荐**：CANN 自定义算子的完整工程（`config/`、`CMakePresets.json`、`build.sh`、两侧 `CMakeLists.txt`）
> 会随 **CANN 版本与芯片型号** 变化，生产环境请用 **msOpGen** 生成对应版本的标准工程，再按本文第 3 节
> 填入 `op_kernel` / `op_host` 的实现代码。本仓库中的 `CMakeLists.txt` / `build.sh` 为参考脚手架。

### 4.1 前置环境
```bash
# 已安装 CANN，并激活环境（按实际 toolkit 路径）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 4.2 用 msOpGen 生成标准工程（推荐）
```bash
# 以 AddCustom 为例（具体参数见 CANN「算子开发」文档，按 CANN 版本调整）
msOpGen --project_name=AddCustom \
        --op_type=AddCustom \
        --op_type_list="add_custom" \
        --soc_version=ascend910b3 \
        --op_api=ascendc \
        --project_path=$(pwd)
# 将本仓库 op_kernel / op_host 的实现覆盖进生成的工程对应文件
```

### 4.3 或直接使用本仓库脚手架编译
```bash
cd add_custom
bash build.sh          # 产物位于 build/kernel 与 build/host
```

### 4.4 功能验证
算子编译为 `libadd_custom.so` 后，可通过 ACL 或 aclnn 接口调用：分配 float16 张量 `x`、`y`
（shape `[N2, N1]`），调用 `add_custom`，核对 `z == x + y`。

## 5. 关键假设与限制（当前为「基础版」）
1. **对齐假设**：`totalLength` 需能被 `NUM_BLOCKS * TILE_NUM * BUFFER_NUM` 整除（即核数、分块、
   双缓冲整除），且每块长度满足向量搬运对齐要求，否则尾部元素可能丢失。基础版未做非对齐（unalign）处理。
2. **单分支**：未使用 `TilingKey` 多分支；如需按 shape/dtype 选择不同核实现，可引入多分支策略
   （`SetTilingKey` + `TILING_KEY_IS`）。
3. **芯片型号**：`op_host/add_custom.cpp` 中 `AddConfig("ascend910b3")` 与 `build.sh` 的
   `SOC_VERSION` 需替换为你实际的芯片型号。
4. **workspace**：本算子未使用需要系统 workspace 的高阶 API，故 workspace 设为 0。

## 6. 可扩展方向
- **多数据类型**：在原型中为输入输出增加 `DataType`/`Format` 组合（如 `DT_FLOAT`、`DT_INT32`），
  核函数中改用 `DTYPE_X`/`DTYPE_Y`/`DTYPE_Z` 宏替代硬编码 `half`，使其随原型自动适配。
- **非对齐 shape**：参考 CANN「动态 / 非对齐 shape」样例，在 TilingData 中增加
  `formerNum / tailNum / formerLength / tailLength / alignNum` 等字段处理尾块。
- **多分支 / 多模板**：依据运行条件用 `TilingKey` 或 TPL/SEL 模板选择不同核实现。

## 7. 常见问题
- **`QuePosition` 找不到 / 编译报错**：部分 CANN 版本使用旧名 `TPosition`（`TQue<TPosition::VECIN, N>`）。
  若目标版本不识别 `QuePosition`，把核函数中的 `QuePosition` 统一替换为 `TPosition` 即可。
- **`GetBlockNum()` 为 0 / 数据没算完**：检查 `NUM_BLOCKS` 与芯片可用 AICore 数、以及 shape 是否满足第 5 节对齐假设。
