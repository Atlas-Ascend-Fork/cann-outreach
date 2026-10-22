# add_custom 算子（Ascend C）

对两个输入张量 `x`、`y` 执行逐元素加法，返回结果张量 `z`：**z = x + y**。

- 形状：2 维 `[N2, N1]`（ND 格式）
- 数据类型：float16（`half`）

## 目录结构

```
add_custom/
├── add_custom.cpp        # Device 侧：核函数 add_custom + KernelAddCustom 类（Add/双缓冲流水）
├── add_custom_tiling.h   # Host/Device 共享的 Tiling 参数结构体
├── add_custom_host.cpp   # Host 侧：aclrt 启动核函数的封装 add_custom_do
├── CMakeLists.txt        # 编译脚本（aic 编译核函数 + host 库）
└── README.md
```

## 实现要点

1. **多核切分**：总元素数 `totalLength = N2 * N1` 按 AI Core 个数均分为 `blockLength`。
2. **核内切块**：每个 Core 再按 `tileLength` 分为 `tileNum` 次迭代，逐块处理。
3. **三级流水**：`CopyIn`（GM→Local，VECIN 队列）→ `Compute`（`Add(z, x, y)` 逐元素加法）→ `CopyOut`（Local→GM，VECOUT 队列），配合 `BUFFER_NUM=2` 双缓冲提高并行度。
4. 约束：`totalLength` 需能被 `blockDim * tileLength` 整除（简化处理，不涉及尾块）。

## 编译

```bash
source ${ASCEND_HOME_PATH}/set_env.sh      # CANN 环境
# 修改 CMakeLists.txt 中的 --soc_version=<SOC_VERSION> 为实际型号（如 Ascend910B1）
mkdir build && cd build
cmake .. && make
```

## 调用示例（Host）

```cpp
// x, y, z 为 device 上的 half* 指针；totalLength = N2 * N1
add_custom_do(8, nullptr, stream, x, y, z, totalLength, 1024);
// blockDim=8 个 Core，每 Core 每次迭代处理 1024 个元素
```
