# Ascend C ReLU 自定义算子

算子名为 `ReluCustom`，计算 `y = max(x, 0)`，用于 Ascend 910B / Atlas A2，目标开发环境为 CANN 8.x。

| 参数 | 输入 x | 输出 y |
| --- | --- | --- |
| 数量 | 1 | 1 |
| Shape | `[N4, N3, N2, N1]` | 与 x 相同 |
| 数据类型 | float16 | float16 |
| Format | ND | ND |

各维度在运行时为非负整数；支持空张量。图编译阶段允许维度为 `-1`，运行时 Tiling 需要具体尺寸。数据按连续 ND 存储，不提供任意 stride 的视图计算。总字节数须不超过 `INT64_MAX`，实际可用尺寸还受设备显存及框架限制。

## 实现文件

- `op_host/relu_custom.cpp`：算子注册、Shape / Dtype 推导和 Tiling 回调。
- `op_host/relu_custom_tiling.h`：Host 到 Kernel 的 Tiling 数据定义。
- `op_host/relu_tiling_policy.h`：独立于 SDK 的实际分片策略，可直接进行 CPU 测试。
- `op_kernel/relu_custom.cpp`：GM → UB → `AscendC::Relu` → GM。
- `relu_custom.json`：msopgen 工程描述。
- `scripts/build.py`：通过本机 CANN 的 msopgen 生成工程、覆盖实现并构建。
- `tests/test_tiling.py`：编译并调用实际 C++ Tiling 策略的 CPU 测试。
- `tests/run_npu.py`：通过生成的 ACLNN 接口执行真机精度和输出保护区检查，无需 NumPy。

## 分片和尾块

ReLU 为逐元素运算，因此将四维输入视作长度为 `N4*N3*N2*N1` 的连续数组。Host 从平台信息获取 AIV 核数及 UB 大小，以 32 字节（16 个 half）为单位均匀分配给各核；仅最后一个核可能处理不足 32 字节的尾块。

每核按 tile 循环执行，输入和输出队列各分配两个 buffer。UB 分配预留 8 KiB，tile 最大为 8192 个 half。核间起点均为 32 字节对齐，彼此不重复写入数据。元素数量和全局偏移使用 64 位整数。

对齐块使用 `DataCopy`。尾块搬入使用 `DataCopyPad`，仅从 GM 读取有效字节，并在 UB 右侧补零至 32 字节对齐；计算后按有效字节数搬出，不将补零部分写回。N1 和总元素数均无需为 16 的倍数。空张量直接返回，不分配 UB、不访问数据。

## 构建

先安装并加载 CANN 8.x Toolkit，准备其要求的 Python 依赖及 `g++`、`make`、`cmake`。例如：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 scripts/build.py --soc Ascend910B1
```

`--soc` 使用目标设备的具体型号，例如 `Ascend910B1` 或 `Ascend910B4`。实现注册的计算单元为 `ascend910b`。脚本生成 `build/ReluCustom/`，完整算子包位于其 `build_out/` 下。

只生成工程时：

```bash
python3 scripts/build.py --soc Ascend910B1 --generate-only
```

工程骨架使用本机版本的 msopgen 生成，避免硬编码不同 CANN 版本的 CMake 模板。后续修改源文件后再次运行脚本，会刷新生成工程中的实现。切换 CANN 版本时，请先移动旧的 `build/ReluCustom/`，再重新生成。

## 安装和调用

在目标机器上选择构建得到的 `.run` 包，将其安装到自定义路径，例如：

```bash
bash build/ReluCustom/build_out/<实际生成的算子包文件名>.run --install-path="$PWD/build/opp"
source build/opp/vendors/customize/bin/set_env.bash
```

安装路径及包名以实际构建产物为准；使用本机生成的安装脚本和环境脚本。`scripts/build.py` 不自动安装算子包。

生成的 ACLNN 头文件为 `aclnn_relu_custom.h`。对于一个连续、四维、ND、float16 的输入 aclTensor 和同 Shape 的输出 aclTensor，调用顺序为：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
// 检查每个接口的返回值。
aclnnReluCustomGetWorkspaceSize(x, y, &workspaceSize, &executor);
// 按 workspaceSize 分配 workspace；为 0 时可传 nullptr。
aclnnReluCustom(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

完整的内存分配、张量创建、调用、同步和释放流程见 `tests/run_npu.py`。应用以所用 CANN 版本实际生成的 ACLNN 头文件为准。

## 验证

CPU 分片测试可在普通机器运行，需 Python 3 和 C++ 编译器：

```bash
python3 -B -m unittest discover -s tests -v
# 编译器未加入 PATH 时：
CXX=/path/to/clang++ python3 -B -m unittest discover -s tests -v
```

测试直接编译生产代码中的 `MakeTilingPlan`，检查数据覆盖完整性、核间无重叠、UB 预算、32 字节边界、空张量、非法 Shape、乘积溢出及大于 2^32 的长度，并测试 500 组随机 Shape。

安装算子包、加载 CANN 和自定义包环境后，在 910B 上执行：

```bash
python3 tests/run_npu.py \
    --op-api build/opp/vendors/customize/op_api/lib/libcust_opapi.so

# 指定一个四维输入：
python3 tests/run_npu.py \
    --op-api build/opp/vendors/customize/op_api/lib/libcust_opapi.so \
    --shape 2 3 5 7
```

默认包含小于 32 字节的输入、对齐和非对齐输入、多 tile 输入、一般四维 Shape 和空张量。数据包括正负值、零、float16 最大有限值、最小正规数及正负无穷；逐元素与 CPU 参考值比较，同时检查输出前后的保护区。NaN、非正规数和有符号零的位模式遵循设备 `AscendC::Relu` 的语义，测试不承诺这些特殊值的跨硬件位级一致性。

本次开发环境验证结果：

- 5 项 CPU 测试全部通过，覆盖上述边界及随机 Shape。
- 本地 CANN 9.1.0-beta.3 的 msopgen 工程生成通过。
- 使用本地 SDK 头文件完成 Host C++ 语法检查；使用 Bisheng 按 `dav-c220-vec` 编译 Kernel，生成 `build/check/relu_custom.o`。
- 当前环境缺少 `cmake`、`make`、`g++`，未完成完整 OPP 包构建；未检测到 NPU，未运行真机测试。CANN 8.x 完整编译及真机测试仍需在目标环境执行。

## 官方接口参考

- [Ascend C Relu API](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0032.html)：逐元素 ReLU、half 类型及 LocalTensor 接口。
- [DataCopyPad API（CANN 8.0.0）](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0265.html)：非对齐搬运、按字节指定有效长度及补齐参数。
