# Challenge 13 · LeakyReLU —— 经典激活算子

## 题目背景

LeakyReLU 是对 ReLU 的经典改进：负半轴不再置零，而是乘以一个小的斜率 α（negative_slope），从而缓解 ReLU 的“神经元死亡”问题，广泛用于判别器、语音与中小规模网络。

给定输入张量 `x` 与属性 `negative_slope`，逐元素计算输出 `y[i] = max(x[i], 0) + negative_slope * min(x[i], 0)`。

本题为开源仓规范化算子开发：基于 Ascend C 在昇腾 NPU 上完成 LeakyReLU 算子的 Tiling 函数（op_host 侧）与核函数实现（op_kernel 侧）。与往期单指令算子不同，本算子是**系列首个带属性的算子**：属性在算子定义（op_def）中注册，由 Tiling 函数读取并写入 tilingData 下发，kernel 侧经 tilingData 取用；kernel 内部通过 `Max/Min/Mul/Add + Duplicate` 复合计算实现，需要管理临时 LocalTensor。

## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `x`：float32 / float16 张量，ND 格式，形状 `(8, 2048)` |
| 属性 | `negative_slope`：float，必选，负半轴斜率 α（参考值 0.01） |
| 输出 | `y`：与 `x` 同 dtype、同形状 |
| 计算 | `y[i] = max(x[i], 0) + negative_slope * min(x[i], 0)` |

## 涉及知识点

本题覆盖 Ascend C 开源仓算子开发的核心知识点：

- **开源仓工程结构**：`op_host`（算子注册 / 形状推导 / Tiling）、`op_kernel`（核函数）、`tests/ut`（单元测试）、`examples`（aclnn 调用示例）
- **算子属性链路（本期新增）**：op_def 中 `this->Attr("negative_slope").FloatAttr(REQUIRED)` 注册 → Tiling 中 `context->GetAttrs()->GetAttrPointer<float>(0)` 读取 → 写入 `LeakyReluTilingData.negativeSlope` 字段 → kernel 侧经 tilingData 使用；aclnn 调用示例中属性作为标量参数传入
- **Tiling 函数**（`LeakyReluTilingFunc`）：获取输入 dtype → `ASCENDC_TPL_SEL_PARAM` 按 tilingKey 选择模板实例 → shape 各维相乘得到 `totalLength` → 设置 `tileNum` → `SetBlockDim` 配置启动核数 → 配置 workspace
- **核函数入口**：`__global__ __aicore__` 声明、`REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT` 解析 tiling 数据
- **Vector 编程范式**：`CopyIn → Compute → CopyOut` 三段式流水线
- **多核数据切分**：`blockLength = totalLength / GetBlockNum()`，`SetGlobalBuffer` 按 `GetBlockIdx()` 设置每核 GM 起始地址
- **队列与内存管理**：`TPipe` / `TQue`、`InitBuffer` 分配 UB、`AllocTensor` / `FreeTensor`、`EnQue` / `DeQue`；`TBuf<TPosition::VECCALC>` 临时 Tensor 的分配与复用
- **复合计算指令**：`Duplicate`（广播标量）、`Max` / `Min`（逐元素取大取小）、`Mul` / `Add`，按 `y = max(x,0) + α·min(x,0)` 分解执行
- **性能优化**：调整 `tileNum`、`BUFFER_NUM = 2` 实现 Double Buffer 双缓冲

## 代码编写

1. 用 AtomGit 账号登录 [CANNJudge 平台](https://cannjudge.cn/home)
2. 加入《代码侦探》小组，完成 [Lesson13_leakyrelu](https://cannjudge.cn/cann_code_detective/challenge13-leakyrelucustom) 答案编写，操作详见 [gitcode 讨论 #287](https://gitcode.com/org/cann/discussions/287)
3. 通过平台验证并提交 PR

特别注意：

- ✅ 通过验证集：提交代码后验证通过显示 "PASS"
- ✅ 提交开源仓：填写邮箱、学校/企业代码缩写（后续作为目录，只支持英文、数字、下划线）、学校/企业全称、个人英文标识（gitcode 用户名去掉 @+下划线+题号，例如"wx234234234_13"），按照指引创建令牌并签署 CLA

## 推荐学习路径

1. 先学习 [Ascend C 算子编程基础（第 2 章）](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development/02_AscendC_basic)，掌握核函数开发与 Vector 编程范式
2. 再学习 [开源仓算子开发（第 6 章）](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.03_operator_development_based_on_opensource_repo.ipynb)，掌握开源仓工程结构与 Tiling 开发
3. 复盘挑战 06（ReLU）的实现，对比思考：引入属性与复合计算后，Tiling 与 kernel 各有哪些新增环节

