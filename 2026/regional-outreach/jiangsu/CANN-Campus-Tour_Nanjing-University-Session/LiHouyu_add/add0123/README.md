# add0123 算子 (Ascend C)

基于 Ascend C 的逐元素加法自定义算子：`z = x + y`

- **shape**: 2 维 `[N2, N1]`（host 侧按展平总量 `N2*N1` 分核，与维度划分无关）
- **dtype**: float16
- **Format**: ND
- **算子类型**: `Add0123`，核函数名 `add0123`

## 目录结构

```
add0123/
├── CMakeLists.txt            # 顶层构建
├── build.sh                  # 编译脚本
├── README.md                 # 本说明
├── op_host/                  # host 侧
│   ├── CMakeLists.txt        # 编译原型注册库与 tiling 库
│   ├── add0123.cpp           # 算子注册 + tiling 策略
│   └── add0123_tiling.h      # host/kernel 共用 tiling 结构
├── op_kernel/                # kernel 侧
│   ├── CMakeLists.txt        # 经 ascendc_library 编译核函数
│   └── add0123.cpp           # 核心实现
└── scripts/
    ├── run_test.sh           # 构建 + 测试一键脚本
    └── test_add0123.py       # Python 单元测试
```

## 实现要点

1. **向量指令**：核心计算为一条 `Add(zLocal, xLocal, yLocal, TILE_LENGTH)`，一次处理整片 128 个 half，无标量循环；尾块用 `DataCopyPad` 任意长度搬运，整片按 `TILE_LENGTH` 计算、按实际长度搬出。
2. **批量向量化分核**：tiling 将展平总量均分到 8 个 AIV 核（`BLOCK_DIM=8`），最后一核兜底余量，任意 shape（含不可整除）均正确。
3. **TQue 标准流水线**：`VECIN`（x/y）+ `VECOUT`（z）双队列，`BUFFER_NUM=2` 双缓冲，GM 搬运与向量计算重叠执行。

关键参数（host 与 kernel 需保持一致）：

| 参数 | 值 | 位置 |
|---|---|---|
| BLOCK_DIM / USE_CORE_NUM | 8 | op_host / op_kernel |
| TILE_LENGTH | 128 | op_kernel |
| BUFFER_NUM | 2 | op_kernel |

## 编译

```bash
# 1. 环境（按实际安装路径）
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash

# 2. 编译（默认 Ascend910B1，可传参指定型号）
cd add0123
bash build.sh Ascend910B1

# 或手动:
mkdir -p build_out && cd build_out
cmake .. -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} -DSOC_VERSION=Ascend910B1
cmake --build . -j16
```

产物（`build_out/`）：

| 产物 | 说明 |
|---|---|
| `libcust_opsproto_rt2.0.so` | 算子原型注册库 |
| `libcust_opmaster_rt2.0.so` | tiling 库 |
| `libadd0123_kernels.so` | kernel 侧动态库（含 `aclrtlaunch_add0123` 拉起接口） |

## 测试

```bash
# 一键: 构建 + 单元测试
bash scripts/run_test.sh Ascend910B1

# 仅单元测试（tiling 分核逻辑 + golden 数学校验）
python3 scripts/test_add0123.py
```

## 部署与调用（NPU）

本工程为简化版（不含 msopgen 生成的 framework/cmake 辅助目录）。如需以自定义算子包方式部署到 `ASCEND_OPP_PATH`，推荐：

```bash
# 用 msopgen 生成完整工程后，将本工程 op_host/op_kernel 实现拷入再编译
${ASCEND_CANN_PACKAGE_PATH}/python/site-packages/bin/msopgen gen -i add0123.json \
    -c ai_core-Ascend910B1 -lan cpp -out Add0123Project
```

直接调用方式：链接 `libadd0123_kernels.so`，include 生成的 `aclrtlaunch_add0123.h`，调用

```cpp
aclrtlaunch_add0123(blockDim, nullptr, stream, x, y, z, workspace, tiling);
```

其中 `x/y/z` 为 device 侧 `uint8_t*` 指针，`tiling` 为 op_host tiling 库产出的 tiling 数据（或自填 `totalLength` 的 TilingData 结构）。
