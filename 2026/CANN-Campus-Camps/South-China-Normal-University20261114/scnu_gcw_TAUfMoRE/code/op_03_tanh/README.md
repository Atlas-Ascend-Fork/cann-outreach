# op_03_tanh

提交团队: 华南师范大学
提交者: 马晓北


## 算子实现介绍

本次 Tanh 算子实现主要包括以下核心设计：

1. **Tiling 切分（Host侧）**：
   - 根据输入 Shape 计算总元素数 `totalLength`。
   - 设定单核内分块数 `tileNum = 8`。
   - 结合多核（Block）数量与双缓冲（BUFFER_NUM=2），自动计算每个 Tiling 块的长度 `tileLength`，实现多核并行与流水线调度。

2. **Kernel 计算逻辑（Device侧）**：
   - **防溢出公式转换**：未直接使用原生公式 `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`，而是转换为等价公式 **`y = 1 - 2 / (exp(2x) + 1)`**，有效避免大数值下 `exp(x)` 计算溢出。
   - **向量指令流水**：依次使用 `Muls`（计算2x）、`Exp`、`Adds`（+1）、`Duplicate`（构造常数2）、`Div`、`Muls`（乘-1）、`Adds`（+1）完成计算。

3. **内存与性能优化**：
   - 采用 `TQue` 双缓冲机制（`BUFFER_NUM=2`）实现 `CopyIn`、`Compute`、`CopyOut` 流水线并行。
   - 使用 `TBuf` 申请临时计算内存（`tempBuffer`），用于存放常数张量，配合 `TPipe` 完成统一内存管理。
