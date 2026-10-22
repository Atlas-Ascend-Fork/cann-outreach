# add_custom 算子（Ascend C）

对两个输入张量 `x`、`y` 执行逐元素加法：`z = x + y`。

- **算子名称**：AddCustom（kernel 入口函数名 `add_custom`）
- **数据类型**：float16
- **数据格式**：ND
- **shape**：2 维 `[N2, N1]`（tiling 约束：`N2*N1` 需能被 `8 * 2 * 256 = 4096` 整除，如 64×64）
- **并行策略**：8 个 AI Core 各处理 1/8 数据，核内双缓冲流水（CopyIn → Compute → CopyOut）

## 目录结构

```
add_custom/
├── add_custom_tiling.h   # host/kernel 共用的 tiling 数据结构
├── kernel/
│   └── add_custom.cpp    # kernel 侧：KernelAdd（TPipe/双缓冲/vec Add 指令）
├── host/
│   └── add_custom.cpp    # host 侧：TilingFunc + InferShape + OpDef 注册
├── main.cpp              # ACL 单算子调用（读 x.bin/y.bin，写 z_out.bin）
├── gen_data.py           # 生成测试数据与 golden 值
├── CMakeLists.txt
└── run.sh                # 一键构建 + 运行 + 校验
```

## 关键实现

**kernel 侧**（`kernel/add_custom.cpp`）：

- `KernelAdd::Init`：按 `GetBlockIdx()` 切分 GlobalTensor，`pipe.InitBuffer` 分配双缓冲 Local 内存；
- `Process`：`tileNum * 2` 次流水迭代；
- `Compute`：调用 Ascend C 内置 `Add(dst, src0, src1, count)` 向量加法指令；
- kernel 入口 `extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)`，通过 `GET_TILING_DATA` 解析 host 下发的 tiling。

**host 侧**（`host/add_custom.cpp`）：

- `TilingFunc`：由输入 shape 计算 `totalLength`、`tileNum`，`SetBlockDim(8)` 后保存到 RawTilingData；
- `OpDef` 注册：输入 `x`、`y`，输出 `z`，均为 `DT_FLOAT16` / `FORMAT_ND`，`AICore().SetTiling` 绑定 tiling 函数。

## 构建与运行

需已安装 CANN toolkit（≥ 7.0，支持 `ascendc_library`），并 source 环境脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash run.sh -v Ascend910B4          # soc 版本按实际硬件调整，如 Ascend310P3 / Ascend910B1
```

或分步执行：

```bash
python3 gen_data.py 64 64           # 生成 x.bin / y.bin / z_golden.bin
mkdir build && cmake -B build && cmake --build build -j8
./build/bin/main 64 64              # 输出 z_out.bin
```

校验脚本会比较 `z_out.bin` 与 `z_golden.bin`，输出 `PASS/FAIL`。

## 自定义 shape

保证 `N2 * N1 % 4096 == 0` 即可（如 32×128、64×64、128×128）。若需要支持任意 shape，需在 tiling 中加入尾块处理逻辑。
