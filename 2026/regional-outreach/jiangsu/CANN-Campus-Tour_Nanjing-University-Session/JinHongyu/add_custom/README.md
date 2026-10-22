# add_custom 自定义算子（Ascend C）

逐元素加法算子：`z = x + y`。

- 算子名：`AddCustom`（opp 包名 `add_custom`，kernel 函数 `add_custom`，aclnn 接口 `aclnnAddCustom` / `aclnnAddCustomGetWorkspaceSize`）
- 输入 `x`、`y`，输出 `z`，均为 **二维 [N2, N1]、float16、ND** 格式
- 目标芯片：**Ascend 910B**（`ascend910b` / `Ascend910B1`）
- 参考工程：本仓库 `JinHongyu/2026/Ascend_C_exercises‌/temple/temple_opensource` 与 `sample/sample_opensource` 的集训营模板（opensource 工程结构）

## 1. 文件清单

```
add_custom/
├── CMakeLists.txt                  # 工程入口（npu_op_package，包名 add_custom，默认 soc=ascend910b）
├── build.sh                        # 编译/打包/UT/示例 一键脚本
├── op_host/                        # host 侧
│   ├── CMakeLists.txt              #   tiling 动态库 + aclnn 接口库 + graph 库
│   ├── add_custom_def.cpp          #   算子原型（float16 / ND / AutoContiguous，ascend910b）
│   ├── add_custom_infershape.cpp   #   shape/dtype 推导（输出 = 输入 x）
│   └── add_custom_tiling.cpp       #   TilingFunc：totalLength、tileLength、blockDim=8
├── op_kernel/                      # device 侧（Ascend C）
│   ├── CMakeLists.txt
│   ├── add_custom.h                #   Kernel 类：多核切分 + 分块 + 尾块 + 双缓冲流水线
│   ├── add_custom.cpp              #   __global__ 入口 add_custom
│   ├── add_custom_tiling_data.h    #   AddTilingData 结构（host/kernel 共用）
│   └── add_custom_tiling_key.h     #   模板参数（仅 C_DT_FLOAT16）
├── examples/                       # aclnn 两段式调用示例（NPU 真机）
│   ├── CMakeLists.txt
│   ├── run.sh
│   └── test_aclnn_add_custom.cpp   #   [8,2048] float16，回读 z 做 MERE/MARE 校验
└── tests/ut/                       # CPU 仿真 UT（tikicpulib，不依赖 NPU 设备）
    ├── CMakeLists.txt
    ├── run.sh
    ├── cmake/BuildGoogleTest.cmake #   源码编译 gtest 1.14（OLD ABI，对齐 CANN 库）
    ├── common/                     # tiling UT 通用基建（context faker / case executor）
    ├── op_host/                    # host 侧 tiling UT（对齐/非对齐两例，校验 tilingKey 与 tilingData）
    └── op_kernel/                  # kernel UT：3 种形状 CPU 仿真 + float16 精度比对
```

## 2. 实现思路

### 2.1 Tiling（host 侧）

host 侧只传两个标量，把切分细节留在 kernel：

| 字段 | 含义 |
|---|---|
| `totalLength` | x 的总元素数（shape 各维相乘） |
| `tileLength` | 单核内每块元素数，固定 1024（float16 下每块 2KB；双缓冲 ×3 队列仅占 12KB，远小于 910B 单核 256KB UB） |

另设 `SetBlockDim(8)`（启动 8 个 AI 核）、workspace = 0。

> 与模板的区别：模板样例把 `blockLength = totalLength / GetBlockNum()`、`tileLength = blockLength / tileNum / BUFFER_NUM` 直接整除，**非对齐形状会静默丢掉余数元素**（如 300 元素 8 核时只处理 256 个）。本实现把余数处理放到 kernel，保证任意形状完整处理。

### 2.2 Kernel（device 侧）

两级切分 + 流水线：

1. **多核切分（余数分摊）**：`base = totalLength / blockNum`，`rem = totalLength % blockNum`；前 `rem` 个核各处理 `base+1` 个元素，其余核处理 `base` 个。每核区间连续、无遗漏、无重叠。
2. **单核内分块**：`tileNum = ceil(blockLen / tileLength)`；第 `i` 块有效长度 `curLen = min(tileLength, blockLen - i*tileLength)` —— 即**尾块边界**：最后一块只搬运/计算 `curLen` 个元素，UB buffer 按 `tileLength` 上限分配、只用前 `curLen`，不越界。
3. **流水线（双缓冲）**：`TQue` 深度 2，`CopyIn(DataCopy GMEM→UB) → Compute(AscendC::Add) → CopyOut(DataCopy UB→GMEME)` 逐块推进，copy 单元与向量单元并行。

特例：`totalLength < blockNum` 时部分核 `blockLen = 0`、`tileNum = 0`，直接空转，安全。

### 2.3 精度

float16 逐元素加法，`AscendC::Add` 直接以 float16 运算。比对标准采用社区阈值：`MERE < 2^-10` 且 `MARE < 10×2^-10`（golden 以 float64 求和后转 float16）。

## 3. 环境假设（版本声明）

本包**未在本机编译/运行**（本机无 CANN），以下为实现所依据的环境假设，远程执行前请确认：

| 项 | 假设 |
|---|---|
| CANN | 8.x 或 9.x（x86_64 Linux，host 库目录自适应 `x86_64-linux/` / `aarch64-linux/` / `host-libs/`） |
| 芯片 | Ascend 910B（`ascend910b`；kernel UT 仿真 `Ascend910B1`）。其他芯片需在 `add_custom_def.cpp` 的 `AddConfig` 中追加 |
| 编译器工具链 | CANN 自带 CANN-C++ 编译器（`find_package(ASC)` 由 `ASCEND_HOME_PATH` 解析） |
| 环境变量 | `ASCEND_HOME_PATH` 已指向 CANN 根目录（脚本缺省时回退 `/usr/local/Ascend/cann`） |
| UT 依赖 | `tools/tikicpulib`（CPU 仿真）可用；gtest 由 CMake 源码编译（需可访问 gitcode 镜像，离线环境请自行替换 `BuildGoogleTest.cmake` 中的 `GTEST_URL`） |
| NPU 示例 | 需要可用 NPU 设备（`npu-smi info`） |

## 4. 编译与测试（远程环境）

### 4.1 编译算子包（编译检查）

```bash
cd add_custom
bash build.sh                 # cmake + 编译 host/kernel + 生成 custom_opp_add_custom_*.run 安装包
bash build.sh --make_clean    # 清理 build/
```

成功标志：输出 `Build completed successfully!`，且存在
`build/op_kernel/ascendc_kernels/binary/ascend910b/*.o` 与 `build/custom_opp_*.run`。

### 4.2 安装算子包（NPU 示例前置）

```bash
bash build/custom_opp_add_custom_*.run --install-path=$ASCEND_HOME_PATH/opp
```

### 4.3 CPU 仿真 UT（正常形状 + 非对齐形状 + float16 误差，不依赖 NPU）

```bash
bash build.sh -u              # 等价：cd tests/ut && ./run.sh
```

覆盖：

| 用例 | 形状 | 元素数 | 验证点 |
|---|---|---|---|
| `AddCustomKernelTest.aligned_8x2048` | [8, 2048] | 16384 | 对齐形状，每核 2 个完整 tile（1024×2） |
| `AddCustomKernelTest.unaligned_cross_core_3x100` | [3, 100] | 300 | 跨核非对齐：300 = 4×38 + 4×37，余数分摊 |
| `AddCustomKernelTest.unaligned_tail_10x1200` | [10, 1200] | 12000 | 核内尾块：每核 1500 = 1024 + 476 |
| `AddCustomTilingTest`（op_host UT） | [8,2048] / [3,100] | 16384 / 300 | tilingKey=0、tilingData、workspace=0 |

每个 kernel 用例回读 `z`，与 golden（float64 求和→float16）做 `MERE/MARE` 比对并打印结果。

### 4.4 NPU 真机端到端（aclnn 两段式调用）

```bash
bash build.sh -e              # 等价：先 build，再 cd examples && ./run.sh
```

`examples/test_aclnn_add_custom.cpp` 以 [8, 2048] float16 ND 调用
`aclnnAddCustomGetWorkspaceSize` + `aclnnAddCustom`，回读 `z` 校验 `z = x + y`
（float16 容差同上）。要测非对齐形状，把 `shape` 改为 `{3, 100}` 重新编译即可。

## 5. 未验证事项（如实声明）

- **本包未在本机编译、未运行任何测试**（本机无 CANN/NPU，且按任务约束不安装环境）。
  上文所有“成功标志”均为预期，需在远程环境按 §4 实际执行确认。
- 未验证项清单：
  1. `npu_op_code_gen` 代码生成对 `OP_ADD(AddCustom)`（多词算子名）的宏展开与 aclnn 接口命名（`aclnnAddCustom*`）是否一致 —— 若远程生成接口名不同，以实际生成的 `aclnn_add_custom.h` 为准调整 examples。
  2. kernel UT 中 `__half` 在 tikicpulib CPU 仿真下的 `AscendC::Add`/`DataCopy` 支持（910B 真机路径由 NPU 编译器直接实例化，风险更低）。
  3. 双缓冲（`BUFFER_NUM=2`）+ 尾块 `curLen` 在真机上的性能与 UB 占用（功能正确性由 UT 覆盖）。
  4. gtest 源码下载地址（gitcode 镜像）在远程环境可达性。
- 若远程 CANN 版本与假设不符（如 7.x、aarch64 设备机），host 库路径自适应逻辑已覆盖 8.x/9.x 常见布局，7.x 未验证。

## 6. 与模板的差异说明

| 点 | 集训营模板（code_template_add） | 本实现 |
|---|---|---|
| 算子名 | 包 `add_custom` / 算子 `Add` | 包与算子统一为 `add_custom` / `AddCustom` |
| 数据类型 | float + float16 | 仅 float16（按任务要求） |
| 非对齐形状 | `totalLength / blockNum` 整除，**丢余数元素** | 多核余数分摊 + 核内尾块 `curLen` 截断，任意形状完整处理 |
| tileLength | 由 tileNum 反推整除 | host 固定 1024，kernel 按 `ceil` 分块 |
| kernel UT | 仅 [8,2048] float32，输出存 bin 由 python 比对 | 3 形状 float16，C++ 内直接 MERE/MARE 比对（无 python 依赖） |
