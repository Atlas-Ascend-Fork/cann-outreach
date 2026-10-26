# Challenge 08 · Tanh —— HansonZhang 提交

本目录是「代码侦探」第八期的个人提交。除补全后的算子实现与官方验证脚本外，额外附带一份设计说明。

## 文件清单

| 文件 | 说明 |
| :--- | :--- |
| `code/challenge08_tanhcustom/op_kernel/tanh.h` | 在模板基础上补全 Tanh 类的 Init/CopyIn/Compute/CopyOut/Process 方法 |
| `code/challenge08_tanhcustom/op_kernel/tanh.cpp` | Kernel 入口，按 tilingKey 分发到 half/float 实例 |
| `code/challenge08_tanhcustom/op_kernel/tanh_tiling_data.h` | 定义 TanhTilingData 结构体 (totalNum, blockFactor, ubFactor) |
| `code/challenge08_tanhcustom/op_kernel/tanh_tiling_key.h` | 定义 tiling 模板参数 schMode (0: fp16/bf16, 1: fp32) |
| `code/challenge08_tanhcustom/op_host/tanh_tiling.cpp` | TilingFunc 实现：计算 blockFactor/ubFactor(32字节对齐)，SetBlockDim，选择 tilingKey |
| `code/challenge08_tanhcustom/op_host/tanh_infershape.cpp` | 形状推导：输出 shape = 输入 shape（模板已完整） |
| `code/challenge08_tanhcustom/op_host/tanh_def.cpp` | 算子定义：输入输出均支持 float32/float16，ND 格式（模板已完整） |
| `code/challenge08_tanhcustom/tests/ut/...` | 官方单测文件，未修改 |
| `code/challenge08_tanhcustom/examples/test_aclnn_tanh.cpp` | 官方验证脚本，未修改 |
| `README.md` | 本文件 |

## 补全思路

题目要求实现逐元素 Tanh 算子：`y = tanh(x)`，输入输出同 dtype 同 shape，支持 float32 和 float16。

### 1. Tiling 数据结构 (tanh_tiling_data.h)
定义三个字段：
- `totalNum`: 总元素数 (8 × 2048 = 16384)
- `blockFactor`: 每个核处理的元素数，按 32 字节对齐
- `ubFactor`: 每次 UB 循环处理的元素数，等于 blockFactor

### 2. Tiling Key (tanh_tiling_key.h)
按数据类型分两个模板实例：
- `schMode = 0`: float16 / bfloat16 (2 字节)
- `schMode = 1`: float32 (4 字节)

### 3. Kernel 类实现 (tanh.h)
- **Init**: 计算当前核的偏移和处理长度，设置 GlobalTensor，初始化输入/输出队列 (BUFFER_NUM=2 双缓冲) 和临时 buffer (VECCALC，大小 `ubLength * sizeof(float) * 2`)
- **CopyIn**: 从 GM 通过 DataCopy 搬运到 LocalTensor，EnQue 入输入队列
- **Compute**: DeQue 获取输入，AllocTensor 分配输出，获取临时 buffer，调用 `AscendC::Tanh<T, false>` 内置高阶 API 计算，EnQue 输出，FreeTensor 输入
- **CopyOut**: DeQue 获取输出，DataCopy 搬回 GM，FreeTensor
- **Process**: 主循环 `CopyIn → Compute → CopyOut`，处理完整块和尾块

### 4. Kernel 入口 (tanh.cpp)
根据 `schMode` 实例化 `Tanh<half>` 或 `Tanh<float>` 并调用 Init/Process。

### 5. TilingFunc (tanh_tiling.cpp)
- 获取平台信息：UB 大小、AIV 核数
- 计算 `totalNum` 从输入 shape
- `blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), 32/typeSize)` 保证 32 字节对齐
- `ubFactor = blockFactor` (每核一次性处理完)
- `SetBlockDim(coreNum)` 启动所有 AIV 核
- 根据 dtype 选择 tilingKey (0: fp16/bf16, 1: fp32)

## 运行验证

在 Ascend 950 环境（如 HiDevLab）的题目目录下执行：

```bash
# 编译并运行 aclnn 验证
cd 2026/CANN-Code-Detective/Challenge08-TanhCustom
mkdir -p build && cd build
cmake .. && make
./test_aclnn_tanh
```

预期输出：程序正常退出，返回 0。

## 模型使用记录

- 实现任务：opencode/nemotron-3-ultra-free (Tier 1 - API 签名核对、算法正确性)
- 无降级发生