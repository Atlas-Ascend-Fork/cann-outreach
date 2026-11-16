# Challenge 14 · Addcmul —— 三输入组合计算算子

## 题目背景

Addcmul 是“加法 + 乘法”的组合算子：`y = input + value * x1 * x2`，在优化器参数更新（如 Adagrad、LARS 的梯度缩放累加）、权重衰减项计算等场景中大量出现，是组合运算类的典型代表。

本题为开源仓规范化算子开发：基于 Ascend C 在昇腾 NPU 上完成 Addcmul 算子的 Tiling 函数（op_host 侧）与核函数实现（op_kernel 侧）。本题是**系列首个三输入算子**：三条 VECIN 队列的 CopyIn 编排、三路 Global Memory 按核切分，加上标量属性 value 的下发，kernel 内通过 `Mul/Mul/Add + Duplicate` 复合计算实现。

## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `input`、`x1`、`x2`：float32 / float16 张量，ND 格式，形状 `(8, 2048)` |
| 属性 | `value`：float，必选，乘积缩放系数（默认参考值 1.0） |
| 输出 | `y`：与输入同 dtype、同形状 |
| 计算 | `y[i] = input[i] + value * x1[i] * x2[i]` |

## 涉及知识点

本题覆盖 Ascend C 开源仓算子开发的核心知识点：

- **开源仓工程结构**：`op_host`（算子注册 / 形状推导 / Tiling）、`op_kernel`（核函数）、`tests/ut`（单元测试）、`examples`（aclnn 调用示例）
- **三输入队列管理（本期新增）**：三条 `TQue<VECIN>` 队列的 InitBuffer 分配、CopyIn 中三次 AllocTensor/DataCopy/EnQue 的编排，Compute 中按序 DeQue 并 FreeTensor
- **算子属性链路**：op_def 中 `this->Attr("value").FloatAttr(REQUIRED)` 注册 → Tiling 中 `context->GetAttrs()->GetAttrPointer<float>(0)` 读取 → 写入 `AddcmulTilingData.value` 字段 → kernel 侧经 tilingData 使用
- **Tiling 函数**（`AddcmulTilingFunc`）：获取输入 dtype → `ASCENDC_TPL_SEL_PARAM` 按 tilingKey 选择模板实例 → shape 各维相乘得到 `totalLength` → 设置 `tileNum` → `SetBlockDim` 配置启动核数 → 配置 workspace
- **核函数入口**：`__global__ __aicore__` 声明、`REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT` 解析 tiling 数据
- **Vector 编程范式**：`CopyIn → Compute → CopyOut` 三段式流水线
- **多核数据切分**：`blockLength = totalLength / GetBlockNum()`，三个输入与一个输出均按 `GetBlockIdx()` 设置每核 GM 起始地址
- **复合计算指令**：`Mul`（x1×x2）、`Duplicate`（value 广播）、`Mul`（×value）、`Add`（+input），两块 `TBuf<VECCALC>` 临时 Tensor
- **性能优化**：调整 `tileNum`、`BUFFER_NUM = 2` 实现 Double Buffer 双缓冲

## 代码编写

1. 用 AtomGit 账号登录 [CANNJudge 平台](https://cannjudge.cn/home)
2. 加入《代码侦探》小组，完成 [Lesson14_addcmul](https://cannjudge.cn/cann_code_detective/challenge14-addcmulcustom) 答案编写，操作详见 [gitcode 讨论 #287](https://gitcode.com/org/cann/discussions/287)
3. 通过平台验证并提交 PR

特别注意：

- ✅ 通过验证集：提交代码后验证通过显示 "PASS"
- ✅ 提交开源仓：填写邮箱、学校/企业代码缩写（后续作为目录，只支持英文、数字、下划线）、学校/企业全称、个人英文标识（gitcode 用户名去掉 @，例如"wx234234234"），按照指引创建令牌并签署 CLA

## 推荐学习路径

1. 先学习 [Ascend C 算子编程基础（第 2 章）](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development/02_AscendC_basic)，掌握核函数开发与 Vector 编程范式
2. 再学习 [开源仓算子开发（第 6 章）](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.03_operator_development_based_on_opensource_repo.ipynb)，掌握开源仓工程结构与 Tiling 开发
3. 复盘代码侦探历史赛题提交记录：对比单输入属性算子，思考三输入时队列分配、GM 切分与 CopyIn 编排各有哪些变化
