# add_custom 算子（Ascend C）

基于 Ascend C 编程语言实现的逐元素加法算子：`z = x + y`。

## 算子规格

| 属性 | 规格 |
|------|------|
| 算子名称 | add_custom |
| 数学公式 | z = x + y |
| 输入/输出 | x, y（输入），z（输出） |
| shape | 2 维 [N2, N1] |
| 数据类型 | float16 (half) |
| 数据格式 | ND |

## 文件结构

```
.
├── add_custom.cpp   # kernel 侧实现（tiling 结构、KernelAddCustom 类、kernel 入口）
├── main.cpp         # host 侧调用程序（ACL 流程：初始化->分配内存->启动 kernel->校验）
├── CMakeLists.txt   # CMake 构建脚本
├── build.sh         # 一键编译运行脚本
└── README.md
```

## 实现要点

- **double buffer 流水**：`TPipe` + `BUFFER_NUM=2`，CopyIn / Compute / CopyOut 三级流水并行。
- **tiling 切分**：单核处理全部数据，`tileLength` 按 32 元素对齐逐倍扩大，保证地址对齐。
- **向量指令**：核心计算使用 `Add(zLocal, xLocal, yLocal, tileLength)` 向量加法指令。

## 编译运行（昇腾环境）

```bash
bash build.sh
```

或在 CANN 环境下手动：

```bash
mkdir build && cd build
cmake .. && make
./add_custom_host
```

运行成功输出 `add_custom PASS`，结果写入 `output/z.bin`。

> 注意：kernel 与 host 的 `AddCustomTilingData` 结构体必须保持一致。
