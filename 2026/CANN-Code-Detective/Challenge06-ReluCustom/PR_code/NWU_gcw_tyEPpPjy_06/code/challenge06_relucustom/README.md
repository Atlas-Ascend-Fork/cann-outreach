# Challenge06 · Relu —— 逐元素 ReLU 激活算子

基于 Ascend C 的开源仓规范化算子实现：

- 输入：`x`，float32 / float16，ND 格式
- 输出：`y = max(0, x)`，与 `x` 同 dtype、同 shape
- op_host：算子注册、形状推导、Tiling
- op_kernel：多核切分、UB 双缓冲、尾块 `DataCopyPad`、`AscendC::Relu` 计算

## 已完成的 TODO

| 文件 | 实现内容 |
| :--- | :--- |
| `op_kernel/relu_tiling_data.h` | 定义 `totalNum / blockFactor / ubFactor` |
| `op_host/relu_tiling.cpp` | 获取平台信息 → 算 shape 总元素数 → 按核切分 → 双缓冲计算 UB 分块大小 → 配置 workspace → 按 dtype 选 tilingKey |
| `op_host/relu_infershape.cpp` | 输出 shape 与输入一致 |
| `op_kernel/relu.cpp` | kernel 入口，解析 tilingData 并调用 `NsRelu::Relu` |
| `op_kernel/relu.h` | `CopyIn → Compute → CopyOut` 流水线，UB 双缓冲 + 尾块处理 |
| `tests/ut/*` | 同步更新 tiling / kernel 单测期望 |

## 编译与自测

```bash
# 默认 ascend910b 编译
bash build.sh

# 跑单元测试
bash build.sh -u
```
