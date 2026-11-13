# 个人开发者

## 团队信息

- 提交者: HansonZhang
- 身份: 个人开发者
- 单位: 个人

## 成员

- HansonZhang (HansonZhang): 提交者

## 本期提交说明

- 期号：代码侦探第 13 期（Challenge 13 · LeakyReLU Custom）
- 算子：逐元素 LeakyReLU 激活算子 `Leakyrelu`，
  `y[i] = max(x[i], 0) + negative_slope * min(x[i], 0)`，
  等价于 `y[i] = (x[i] > 0) ? x[i] : negative_slope * x[i]`
- 目录：`code/exercises/`（内容为官方初始模板工程 + 本人补全的 4 个文件）
- ★ 本期特色：**首次引入算子属性 `negative_slope`**，
  下发链路为 `op_host/leakyrelu_def.cpp`（注册属性）
  → `op_host/leakyrelu_tiling.cpp`（`context->GetAttrs()->GetAttrPointer<float>(0)` 读出）
  → `op_kernel/leakyrelu_tiling_data.h`（`LeakyreluTilingData::negativeSlope`）
  → `op_kernel/leakyrelu.h`（`Init()` 读出、`Compute()` 使用）
- 补全范围（4 个文件）：
  - `op_kernel/leakyrelu_tiling_data.h`：新增 `float negativeSlope` 属性字段
  - `op_kernel/leakyrelu.h`：`Init` / `CopyIn` / `Compute` / `CopyOut` / `Process` 全部实现
  - `op_host/leakyrelu_tiling.cpp`：totalNum / blockFactor / ubFactor / blockDim 计算 +
    属性读取与下发 + tilingKey 选择
  - `op_host/leakyrelu_infershape.cpp`：仅补 TODO 说明（模板自带写法已正确，未重写）
- 官方单测（`tests/ut/**`）、`examples/**`、`samples/**`、构建脚本、
  `op_host/leakyrelu_def.cpp`、`op_kernel/leakyrelu.cpp`、`op_kernel/leakyrelu_tiling_key.h`
  **逐字节未改**
- 输入输出：x / y，shape (8, 2048)，支持 float32 / float16，ND 格式
- 验证：见 PR 正文「验证说明」一节（**本机无 NPU，未做实机验证，如实说明**）
