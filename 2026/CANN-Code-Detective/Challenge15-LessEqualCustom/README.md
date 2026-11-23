# Challenge 15 · LessEqual —— bool 输出的比较算子

## 题目背景

比较算子是掩码生成、条件筛选、损失掩码等场景的基础构件：`y = (x1 <= x2)` 逐元素比较并输出 bool 掩码，广泛用于注意力掩码、梯度截断与数据过滤。

本题为开源仓规范化算子开发：基于 Ascend C 在昇腾 NPU 上完成 LessEqual 算子的 Tiling 函数（op_host 侧）与核函数实现（op_kernel 侧）。本题是**系列首个输出类型与输入不同的算子**：输出为 bool（uint8，1 字节/元素），输出队列需按 1 字节分配；kernel 侧通过 `Compare + Select` 数据通路完成比较与选择，并管理掩码与真/假值三块临时 Tensor。

## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `x1`、`x2`：float32 / float16 张量，ND 格式，形状 `(8, 2048)` |
| 输出 | `y`：bool 张量（1 字节/元素），形状 `(8, 2048)` |
| 计算 | `y[i] = (x1[i] <= x2[i]) ? 1 : 0` |

## 涉及知识点

本题覆盖 Ascend C 开源仓算子开发的核心知识点：

- **开源仓工程结构**：`op_host`（算子注册 / 形状推导 / Tiling）、`op_kernel`（核函数）、`tests/ut`（单元测试）、`examples`（aclnn 调用示例）
- **异构输出（本期新增）**：op_def 中输出注册为 `ge::DT_BOOL`；输出队列 `InitBuffer` 按 `sizeof(uint8_t)` 分配（1 字节/元素，与输入 dtype 解耦）；输出 `GlobalTensor<uint8_t>` 的 GM 切分
- **比较与选择指令（本期新增）**：`AscendC::Compare`（`CMPMODE::LE` 生成掩码）+ `AscendC::Select`（`SELMODE::VSEL` 按掩码选择）+ `Duplicate` 构造真/假值向量
- **Tiling 函数**（`LessEqualTilingFunc`）：获取输入 dtype → `ASCENDC_TPL_SEL_PARAM` 按 tilingKey 选择模板实例 → shape 各维相乘得到 `totalLength` → 设置 `tileNum` → `SetBlockDim` 配置启动核数 → 配置 workspace
- **核函数入口**：`__global__ __aicore__` 声明、`REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT` 解析 tiling 数据
- **Vector 编程范式**：`CopyIn → Compute → CopyOut` 三段式流水线，双输入队列编排
- **多核数据切分**：`blockLength = totalLength / GetBlockNum()`，`SetGlobalBuffer` 按 `GetBlockIdx()` 设置每核 GM 起始地址
- **队列与内存管理**：`TPipe` / `TQue`、`InitBuffer` 分配 UB、`AllocTensor` / `FreeTensor`、`EnQue` / `DeQue`；三块 `TBuf<VECCALC>` 临时 Tensor（mask/true/false）
- **性能优化**：调整 `tileNum`、`BUFFER_NUM = 2` 实现 Double Buffer 双缓冲

## 代码编写

1. 用 AtomGit 账号登录 [CANNJudge 平台](https://cannjudge.cn/home)
2. 加入《代码侦探》小组，完成 [Lesson15_lessequal](https://cannjudge.cn/cann_code_detective/challenge15-lessequalcustom) 答案编写，操作详见 [gitcode 讨论 #287](https://gitcode.com/org/cann/discussions/287)
3. 通过平台验证并提交 PR

特别注意：

- ✅ 通过验证集：提交代码后验证通过显示 "PASS"
- ✅ 提交开源仓：填写邮箱、学校/企业代码缩写（后续作为目录，只支持英文、数字、下划线）、学校/企业全称、个人英文标识（gitcode 用户名去掉 @，例如"wx234234234"），按照指引创建令牌并签署 CLA

## 推荐学习路径

1. 先学习 [Ascend C 算子编程基础（第 2 章）](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development/02_AscendC_basic)，掌握核函数开发与 Vector 编程范式
2. 再学习 [开源仓算子开发（第 6 章）](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.03_operator_development_based_on_opensource_repo.ipynb)，掌握开源仓工程结构与 Tiling 开发
3. 复盘挑战 14（Addcmul）：同为多输入算子，对比思考输出类型与输入不同时，输出队列的 InitBuffer、GM 切分与 DataCopy 各需要注意什么
