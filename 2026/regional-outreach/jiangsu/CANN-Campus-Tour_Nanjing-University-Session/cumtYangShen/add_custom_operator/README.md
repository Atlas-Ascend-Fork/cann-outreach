# add_custom 算子

基于 **Ascend C**（矢量编程范式）实现的逐元素加法算子。

## 算子规格

| 项 | 规格 |
|---|---|
| 名称 | `add_custom` |
| 功能 | 对两个输入张量执行逐元素加法 |
| 数学公式 | `z = x + y` |
| 形状 | 2 维 `[N2, N1]`，支持任意尺寸（含非对齐尺寸） |
| 数据类型 | `float16`（`half`） |
| 数据格式 | `ND`（2 维下为行主序连续排布，元素总数 = N2 × N1） |
| 内核签名 | `extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t n2, uint32_t n1)` |

## 工程结构

```
add_custom_operator/
├── src/
│   └── add_custom.cpp      # Ascend C 算子内核（核心交付物）
├── test/
│   └── main.cpp            # 主机端测试（数据生成 + ACL 拷贝 + 逐元素校验）
├── build.sh                # 构建脚本（ccec_compiler + g++）
└── README.md
```

## 实现说明

采用 Ascend C **矢量编程范式**，`TPipe` 统一管理 UB 内存，`TQue` 队列完成 GM↔UB 搬运与计算间的同步：

```
Stage1 CopyIn   : DataCopy  GM(x,y) -> UB          （VECIN 队列 inQueueX / inQueueY）
Stage2 Compute  : Add(zLocal, xLocal, yLocal, len) （UB 内向量加法）
Stage3 CopyOut  : DataCopy  UB(z) -> GM            （VECOUT 队列 outQueueZ）
```

**数据切分（两级）：**

1. **核间**：按 `GetBlockNum()` 均分全部元素（向上取整），每个核通过 `SetGlobalBuffer` 只操作自己负责的分片，末核处理余数；
2. **核内**：按 `TILE_LEN = 256` 个 half（512 字节，32 字节对齐的整数倍）分块流水处理，末尾不足一个 tile 的部分按实际长度处理。

因此**任意 `[N2, N1]` 组合（包括元素总数不能被 tile 整除、核数不能整除总数）结果均正确**。

## 构建与运行

> ⚠️ 必须在 **Linux + CANN** 环境（如 Atlas 800I/300I 服务器或 DevKit 的 Ubuntu 环境）中构建运行；Windows 上无法编译 Ascend C 内核。

```bash
# 1. 加载 CANN 环境（示例路径，按实际安装位置调整）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.bash
# 若 set_env.bash 不生效，可手动： export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 2. 构建（默认 Ascend910B，按设备修改）
cd add_custom_operator
SOC=Ascend910B ./build.sh

# 3. 运行测试
cd build && ./add_custom_test
```

预期输出：

```
[test] shape=[4, 10363], dtype=float16, format=ND, total=41452 elements
[test] PASS: add_custom 全部 41452 元素正确（maxErr=0.003906, tol=0.05）
```

## 测试程序启动算子的对接说明

`test/main.cpp` 中数据生成、设备内存管理、结果校验均已完成；**算子启动部分**用 `【CANN 版本相关】` 标出，因为自定义内核的启动 API 随 CANN 版本而异：

- **CANN 7.0+**：使用 `aclrtLoadKernel` + `aclrtLaunchKernel` 加载 `build/add_custom.o` 并启动，参数与内核签名 `(x, y, z, n2, n1)` 一一对应，`blockDim` 核数按设备填写（如 910B 为 48）；
- **自定义算子注册机制**：若走 JSON/tik 注册流程，则通过注册框架的调用接口启动，参数同上。

请按你的 CANN 版本在 `test/main.cpp` 第 4 步补全对应启动代码。

## 注意事项

- 内核中 `TILE_LEN`、队列深度（当前为 1，单缓冲顺序执行）可按性能需求调整：增大 tile 或改双缓冲（`BUFFER_NUM=2` + 跨 tile 流水）可进一步提升吞吐；
- `Add` 为 half 精度向量加法，精度特性与 NPU 上其他 fp16 算子一致；
- 若后续需要支持 float32 / int8 等其他 dtype，只需替换 `half` 为对应类型（`GlobalTensor<T>` / `LocalTensor<T>` / `AllocTensor<T>` / `DeQue<T>` 中的 T），切分逻辑不变。
