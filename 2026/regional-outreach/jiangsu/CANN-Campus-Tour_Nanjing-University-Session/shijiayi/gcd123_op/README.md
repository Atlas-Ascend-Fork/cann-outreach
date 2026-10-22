# Gcd123 —— AscendC 自定义算子（求最大公约数）

基于 **AscendC** 框架实现的 `gcd123` 算子：对两个 float16 张量逐元素求**最大公约数（GCD）**，
支持 **ND 4 维**形状与 **broadcast**。

---

## 1. 算子规格

| 项目 | 说明 |
| --- | --- |
| 算子类型名 | `Gcd123`（kernel 入口名 `gcd123`，遵循大驼峰→小写下划线规则） |
| 输入 | `self`（float16, ND）、`other`（float16, ND） |
| 输出 | `out`（float16, ND） |
| 形状 | 4 维 `[N4, N3, N2, N1]`，`N1` 为连续维 |
| broadcast | `self` 与 `other` 需满足标准 broadcast（逐维相等或其中一方为 1，右对齐） |
| 输出形状 | `out` = `broadcast(self, other)` 的形状 |
| 数据类型 | float16（half） |
| Format | ND |

### 计算语义（重要假设）

GCD 本质是**整数**运算，而输入是 float16。本算子的约定为：

1. 将 `self`、`other` 的元素**截断取整**（向负无穷，`CAST_FLOOR`）得到整数；
2. 对两个整数的**绝对值**做欧几里得辗转相除求 GCD；
3. 将 GCD 结果转回 float16 写入 `out`。

因此当输入是整数值（如 `6.0`、`8.0`）时，结果即为整数 GCD（如 `2.0`）；
`gcd(0, x) = x`、`gcd(x, 0) = x`、`gcd(0, 0) = 0`，负数取绝对值后计算。

> 若你希望“取整”方式不同（如四舍五入），修改 kernel 中 `HalfToI32` 使用的
> `RoundMode`（`CAST_FLOOR` → `CAST_ROUND` 等）即可。

---

## 2. 目录结构

```
gcd123_op/
├── gcd123.json                 # 算子原型定义（msOpGen 输入，2 输入 1 输出 / float16 / ND）
├── build.sh                    # 构建脚本
├── op_host/
│   └── gcd123.cpp              # Host 侧：InferShape(broadcast)、InferDataType、Tiling
├── op_kernel/
│   ├── gcd123_tiling.h         # TilingData（Host 与 Kernel 共享）
│   └── gcd123.cpp              # AscendC Kernel：GCD + broadcast + 块级 DataCopy
└── test/
    ├── gcd123_cpu_test.cpp     # 独立 CPU 参考 + 断言（校验 broadcast/GCD 逻辑）
    └── gcd123_npu_test.cpp     # NPU aclnn 单算子调用骨架（需先编译安装算子）
```

---

## 3. 构建与部署（推荐：msOpGen 生成骨架）

`namespace ops` 的算子原型注册由 **msOpGen** 依据 `gcd123.json` 自动生成，且随 CANN 版本不同，
最稳妥的流程是：先生成骨架，再覆盖实现文件。

```bash
# 0) 每次新终端先加载 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh     # 路径按实际安装调整

# 1) 用 msOpGen 生成工程骨架（-c 取目标芯片 soc_version，如 ai_core-ascend910b）
msopgen gen -i gcd123.json -c ai_core-ascend910b -lan cpp -out Gcd123

# 2) 用本仓库的实现覆盖骨架对应文件：
#      Gcd123/op_kernel/gcd123_tiling.h  <-  op_kernel/gcd123_tiling.h
#      Gcd123/op_kernel/gcd123.cpp       <-  op_kernel/gcd123.cpp
#    并把本仓库 op_host/gcd123.cpp 中的 namespace ge / namespace optiling
#    粘贴进生成的 Gcd123/op_host/gcd123.cpp（保留其自动生成的 namespace ops）。

# 3) 编译
cd Gcd123 && ./build.sh

# 4) 安装算子包（.run 会装到 $ASCEND_OPP_PATH/vendors/customize/）
./build_out/custom_opp_*.run

# 5) 配置单算子 API 库路径
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
```

> `build.sh` 中 `ASCEND_COMPUTE_UNIT` 需与目标芯片一致（如 `ascend910b`/`ascend310p3`）。
> 若使用 msOpGen 生成的 `CMakePresets.json`，可直接 `cmake --preset <preset> && cmake --build build`。

---

## 4. 测试

### CPU 参考测试（无需 CANN，任意平台）

用于校验 **broadcast 形状推导** 与 **GCD 数值** 的正确性，逻辑与 kernel 完全一致：

```bash
g++ -std=c++17 -O2 test/gcd123_cpu_test.cpp -o gcd123_cpu_test
./gcd123_cpu_test        # 期望输出 “ALL PASS”
```

> 说明：当前开发机未安装 C/C++ 编译器，本文件尚未在此实际运行；请在任意装有标准 C++ 编译器
> 的机器上执行。它不依赖 CANN，可直接编译。

### NPU 测试（需先编译安装算子）

`test/gcd123_npu_test.cpp` 给出 aclnn 单算子调用的标准骨架（ACL 初始化、设备内存、取回比对）。
其中 `aclnnGcd123*` 的**接口名与参数签名**以安装后 `op_api` 目录下生成的头文件为准（随版本略有差异），
请对齐后填入注释处的调用。

---

## 5. 实现要点

- **Kernel（`op_kernel/gcd123.cpp`）**
  - 输出按“行”组织（固定 `N4,N3,N2`，沿 `N1` 连续）；每行按 `chunkLen`（默认 1024）分块，
    用块级 `DataCopy` 在 GM↔UB 之间搬运，UB 占用可控。
  - broadcast：`self/other` 的广播维（size=1）取 index 0；内维 `N1` 被广播时整行只取一次并复用。
  - 逐元素 GCD 用**标量(Scalar)** 循环在 UB 上完成；用 `SetFlag/WaitFlag`
    （`MTE2_S`、`S_MTE3`）保证 GM↔UB 搬运与标量读写之间的数据依赖。
  - 多核：把 `totalLines` 行均匀切分给各 AI Core（含余数处理）。
- **Host（`op_host/gcd123.cpp`）**
  - `InferShape`：右对齐 broadcast 推导输出形状；不兼容则报错。
  - `TilingFunc`：写入 4 维形状与派生量（`totalSelf/Other/Out`、`totalLines`、`chunkLen`）。
- **类型转换（kernel）**
  - `float16 → float → int32`：float16 到位级转换（`HalfToFloat`，不依赖隐式转换运算符，跨版本可移植），
    再用 `AscendC::Cast<float,int32_t,CAST_FLOOR>`。
  - `int32 → float → float16`：`AscendC::Cast<float,half,CAST_ODD>`（整数结果转换精确）。

---

## 6. 约束与注意事项

- **元素总数**需能放入 `uint32_t`（< 4.29e9）；更大张量需把 TilingData 与偏移扩为 64 位。
- **数值范围**：float16 可表示整数最大约 65504；GCD 结果 ≤ max(|self|,|other|)，不会溢出。
  非整数值按“截断取整”处理（见 §1）。
- **API 版本**：`REGISTER_TILING_DEFAULT / GET_TILING_DATA`、`GlobalTensor::operator[]`、
  `SetFlag/WaitFlag`、GE 的 `InferShape/InferDataType` 及 `namespace ops` 注册均基于 **CANN 9.x**
  的 msOpGen 模板；若你的 CANN 版本不同，以对应版本 msOpGen 生成物为准微调。
- **AI Core 数**：`GCD123_NUM_BLOCKS`（默认 20）请按目标芯片核数调整。
- **自动同步**：若工程关闭了自动同步，kernel 中已手动插入的 `SetFlag/WaitFlag` 生效；
  若开启自动同步，这些屏障为冗余但无害。

---

## 7. 可进一步优化方向

- GCD 核心是数据相关循环，难以向量化；若性能敏感，可考虑：
  - 提高 `chunkLen` 或采用双缓冲（`TQue` 深度 2）重叠搬运与计算；
  - 对常见“两输入同形状且无内维广播”的主路径做专门快路径（连续块级搬运 + 标量计算）。
- 若需要支持更多 dtype（如 int32 输入直接做 GCD），可用模板入口 + Tiling 模板参数扩展。
