# add_custom 算子

基于 **Ascend C** 编程语言实现的逐元素加法算子。

## 算子定义

| 属性 | 说明 |
| ---- | ---- |
| 算子名称 | add_custom |
| 功能 | 逐元素加法 `z = x + y` |
| 输入/输出 | x, y, z 均为张量 |
| shape | 2 维 `[N2, N1]` |
| 数据类型 | float16（half） |
| 数据格式 | ND |

## 工程结构

```
add_custom/
├── add_custom.cpp           # 核函数：KernelAddCustom（CopyIn→Compute→CopyOut 双缓冲流水）
├── op_host/
│   ├── add_custom_host.cpp  # 宿主侧：tiling 计算、shape/dtype 推导、算子注册
│   └── add_custom_tiling.h  # tiling 结构定义（totalLength, tileLength）
├── scripts/
│   ├── gen_data.py          # 生成 x/y/golden 测试数据
│   └── verify.py            # CPU 模式运行并比对结果
├── CMakeLists.txt
├── build.sh
└── README.md
```

## 实现要点

- **编程范式**：矢量编程（Vector），`TPipe` + `TQue` 管理内存
- **双缓冲**：`BUFFER_NUM = 2`，CopyIn/Compute/CopyOut 流水并行，隐藏数据搬运时延
- **多核切分**：`totalLength` 按核数均分，每核再按 `tileLength`（默认 8192 个 half）分片循环
- **对齐**：tileLength 按 16 元素（32B）对齐，满足 DataCopy 对齐要求
- **tiling 传递**：宿主侧 `TilingFunc` 计算 `[totalLength, tileLength]`，核函数通过 tiling 地址读取

## 构建与运行

```bash
# 1. 准备环境（已安装 CANN 包）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 生成测试数据
python3 scripts/gen_data.py

# 3. 构建算子（NPU 模式，默认 Ascend910B1）
bash build.sh -v Ascend910B1

# CPU 调试模式（无需硬件）
RUN_MODE=cpu bash build.sh

# 4. 验证
python3 scripts/verify.py
```

## 可调整项

- `scripts/gen_data.py` 中 `N2, N1` 改变测试 shape
- `op_host/add_custom_host.cpp` 中 `ELEMENTS_PER_TILE` 调整每次循环处理量
- 核数上限 `40` 按实际 SoC 修改（如 Atlas A2 为 48）
