# add_custom 算子（Ascend C）

对两个输入张量 `x`、`y` 逐元素相加，输出 `z`：

```
z = x + y
```

| 项目 | 取值 |
| --- | --- |
| 算子类型 (Op Type) | `AddCustom` |
| 核函数 (Kernel) 入口 | `add_custom` |
| shape | 2 维，`[N2, N1]` |
| dtype | `float16` |
| format | `ND` |
| 编程范式 | 向量：CopyIn → Compute → CopyOut（双缓冲流水线） |

---

## 1. 交付文件（本次编写）

```
add_custom/
├── add_custom.json              # 算子原型定义（喂给 msopgen 生成工程）
├── op_kernel/
│   ├── add_custom.cpp           # 核函数（Kernel）实现：z = x + y
│   └── add_custom_tiling.h      # Tiling 结构体（Host/Kernel 共享）
├── op_host/
│   └── add_custom.cpp           # Host 侧：TilingFunc + 算子原型注册 (OpDef)
└── README.md                    # 本文件（编译 / 部署 / 验证流程）
```

| 文件 | 作用 |
| --- | --- |
| `op_kernel/add_custom.cpp` | 核函数：`GM(x/y)` →`DataCopy`→ `UB` →`Add`→ `UB(z)` →`DataCopy`→ `GM(z)`。按 AI Core 切块（`GetBlockNum`/`GetBlockIdx`），块内再按 `TILE_NUM` 切 tile，双缓冲重叠 DMA 与计算。 |
| `op_kernel/add_custom_tiling.h` | 定义 `AddCustomTilingData { totalLength; tileNum; }`，Host 写入、Kernel 读取，二进制直接拷贝，两侧字段必须一致。 |
| `op_host/add_custom.cpp` | `TilingFunc` 计算并写入 `totalLength`（= `N2*N1`）、`tileNum`，设置 `SetSimdNumBlocks`；`OpDef` 注册输入/输出（float16 + ND）、shape/dtype 推导、关联 Tiling。 |
| `add_custom.json` | 算子原型定义（IR 形式），`msopgen` 据此生成整个算子工程骨架。 |

> 说明：`op_kernel/`、`op_host/` 这两个目录及其中三个源码文件，正是 `msopgen` 生成工程时对应的目录/文件。用本交付的文件覆盖 msopgen 生成的模板即可。

---

## 2. 编译命令

**前提**：昇腾 AI 处理器开发机，已安装 CANN（含 `msopgen`），并 `source` 过 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 路径按实际版本替换
```

### (1) 生成算子工程骨架

msopgen 读取算子原型定义，生成 `build.sh` / `CMakeLists.txt` / `cmake/` / `scripts/` / `op_kernel/` / `op_host/` 等完整工程：

```bash
# 方式 A：msopgen 在 PATH 中
msopgen gen -i add_custom.json -c ai_core-<soc_version> -lan cpp -out ./AddCustom

# 方式 B：使用 CANN 安装目录下的绝对路径（${INSTALL_DIR} 为 CANN 安装目录，
# 如 /usr/local/Ascend/ascend-toolkit/latest）
${INSTALL_DIR}/python/site-packages/bin/msopgen gen \
    -i add_custom.json -c ai_core-<soc_version> -lan cpp -out ./AddCustom
```

- `-i`  算子原型定义 json 路径
- `-c`  计算资源，格式 `ai_core-<soc_version>`；`<soc_version>` 为芯片型号，如
  `Ascend910B` / `Ascend910` / `Ascend310P` 等（可在 CANN 安装目录
  `compiler/data/platform_config/` 下的 `*.ini` 文件名中查看）。
- `-lan cpp`  用 C++（Ascend C）实现
- `-out`  工程输出目录

> 如需适配特定框架（TF / PyTorch / MS 等），加 `-f tf` / `-f pytorch` / `-f ms`；纯 CANN 自定义算子可不加。

### (2) 用本次交付的源码覆盖模板

```bash
cp op_kernel/add_custom.cpp       AddCustom/op_kernel/add_custom.cpp
cp op_kernel/add_custom_tiling.h  AddCustom/op_kernel/add_custom_tiling.h
cp op_host/add_custom.cpp         AddCustom/op_host/add_custom.cpp
```

### (3) 编译

```bash
cd AddCustom
bash build.sh        # 等价于 cmake --preset=default && cmake --build . --target package -j16
```

产物：`build_out/custom_opp_<os>_<arch>.run`（自解包安装脚本）及算子二进制。

---

## 3. 部署

```bash
# 安装到 CANN 默认路径（安装后需 source set_env.sh 使 ASCEND_CUSTOM_OPP_PATH 生效）
./build_out/custom_opp_*.run

# 或安装到自定义路径
./build_out/custom_opp_*.run --install-path="/xxx/AddCustom/installed"
# 之后将该路径加入 ASCEND_CUSTOM_OPP_PATH：
# export ASCEND_CUSTOM_OPP_PATH=/xxx/AddCustom/installed:$ASCEND_CUSTOM_OPP_PATH
```

---

## 4. 运行验证（ST 测试，可选）

```bash
# 生成 ST 测试用例
msopst create -i "AddCustom/op_host/add_custom.cpp" -out ./st

# 配置环境变量（${INSTALL_DIR} 为 CANN 安装目录）
export DDK_PATH=${INSTALL_DIR}
export NPU_HOST_LIB=${INSTALL_DIR}/${arch-os}/devlib

# 运行 ST 测试
msopst run -i ./st/AddCustom.json -soc <soc_version> -out ./st/out
```

---

## 5. 注意事项

1. **切分整除性**：当前 tiling 采用 `8 core × 8 tile × 2(双缓冲) = 128` 的固定切分，
   要求总元素数 `N2 * N1` 能被 128 整除（官方示例 `[8, 2048]`，`16384 / 128 = 128`，满足）。
   若使用不能被 128 整除的 shape，需扩展 `TilingFunc` 与核函数做尾块（tail）处理。
2. **核心数 / tile 数**：`NUM_BLOCKS`、`TILE_NUM` 定义在 `op_host/add_custom.cpp`，
   可按芯片实际 AI Core 数与数据规模调整。
3. **CANN 版本差异**：本文按 CANN 8.x / 9.x 写法。旧版本中 `SetSimdNumBlocks`
   对应 `SetBlockDim`、`GetTilingData<T>()` 对应 `GetRawTilingData()->GetData()` +
   `SetDataSize()`；核函数侧 `GET_TILING_DATA(tilingData, tiling)` 宏与本文的
   `reinterpret_cast` 等价。若目标 CANN 版本较旧，按上述对照微调即可。
4. **验证状态**：本环境（Windows，无 CANN / NPU）无法实际编译，以上命令需在昇腾开发机执行。
