# CANN训练营 · SoftmaxCustom —— RegBase（reg 矢量编程）行 Softmax 算子

题目目录（相对于 `cann-outreach` 仓库根目录）：`2026/CANN-Code-Detective/CANN-Camps/20261015/`。

## 题目背景

前面几期题目走的是 **Ascend C 的 SIMD / MemBase 编程范式**：用 `TPipe` + `TQue` 管理 UB，`CopyIn → Compute → CopyOut` 三段式流水，计算主体围绕 `LocalTensor` 展开（`DataCopy` / `Exp` / `Div` 等）。上一期（20261008）切换到了 **SIMT 编程范式**：`__global__` + `threadIdx.x` / `blockIdx.x`，`<<<gridDim, blockDim, dynUB, stream>>>` 启动。

本题进入第三条路线：**RegBase（reg 矢量编程）**。它的特点是——

- 核函数以 `__global__ __vector__` 声明，用 `<<<blocks, 0, stream>>>` **直接启动**，不需要 `TPipe` / `TQue` / `LocalTensor`；
- 计算主体写成 `__simd_vf__` **向量函数**，直接在**向量寄存器**上运算：`asc_loadalign` 把 UB 数据搬进寄存器，`asc_max` / `asc_sub` / `asc_exp` / `asc_div` 在寄存器之间直接算，`asc_reduce_max` / `asc_reduce_sum` 做行内归约，`asc_storealign` 把结果写回 UB。

本题对 `81920 × 128` 的 `float32` 张量**按最后一维做 softmax**（每一行独立归一化）。难点在于把 softmax 的三步算法（求行最大值 → 广播相减防溢出 → 指数求和归一化）**拆成寄存器级的指令序列**，并处理好一行长度 `WIDTH = 128` 与"单条向量指令一次能处理多少元素"之间的关系。

## 算子描述

$$\text{Softmax}(x_{i,j}) = \frac{e^{\,x_{i,j} - \max_j x_{i,j}}}{\sum_j e^{\,x_{i,j} - \max_j x_{i,j}}}$$

| 项目 | 说明 |
| :--- | :--- |
| 算子名称 | `SoftmaxCustom` |
| 输入 | `x`：float32 张量，形状 `[81920, 128]`，ND |
| 输出 | `y`：float32 张量，形状 `[81920, 128]`，ND |
| 计算 | 对最后一维（长度 `WIDTH = 128`）做 softmax，每行独立 |
| 输入数据 | `[0, 1)` 均匀分布随机值，host 侧生成后同时用于 kernel 与 golden |
| 数据切分 | 单核（`blocks_num = 1`），单核循环 `LOOP_COUNT = 1024` 次，每次处理 `SINGLE_VF_HEIGHT = 80` 行 |
| 精度标准 | `abs_err < 1e-4` |
| 是否要求多核 / 性能优化 | **不要求**。本题只考 reg 矢量编程本身，把结果算对即可；多核切分、双发、指令融合等优化不作为考点 |

## 答题准备

进入 [Hidevlab](https://hidevlab.huawei.com/online-develop-intro)，申请 **Ascend 950PR / Ascend 950DT** 算力（NPU 架构 `dav-3510`）；申请中注意备注「CANN训练营AscendC950作业算力申请」，审批通过后即可在在线 WebIDE 中开发。

进入环境后拉取题库仓库：

```bash
git clone https://gitcode.com/cann/cann-outreach.git
```

开始答题前，先从 `cann-outreach` 仓库根目录进入题目目录，并将模板复制到以本人 GitCode 用户名命名的个人目录。请将 `your_gitcode_name` 换成自己的用户名：

```bash
cd 2026/CANN-Code-Detective/CANN-Camps/20261015
cp -r Template/SoftmaxCustom your_gitcode_name/SoftmaxCustom
```

后续请在 `your_gitcode_name/SoftmaxCustom/softmax_custom.asc` 中补全代码。

## 待补全任务

代码骨架位于 [Template/SoftmaxCustom/](./Template/SoftmaxCustom/) 目录，其中核函数与 host 侧代码都已给出，**只需要补全 VF 函数 `softmax_vf()` 的函数体**：

- [softmax_custom.asc](./Template/SoftmaxCustom/softmax_custom.asc) —— 算子实现骨架，`softmax_vf()` 待补全；
- [CMakeLists.txt](./Template/SoftmaxCustom/CMakeLists.txt) —— 构建脚本，无需修改；
- [run.sh](./Template/SoftmaxCustom/run.sh) —— 编译运行脚本，一般无需修改。

`softmax_vf()` 内对**每一行**依次完成以下 7 个步骤（在寄存器上完成）：

| 步骤 | 操作 | 可用的寄存器接口（示意） |
| :--: | :--- | :--- |
| 1 | 求每行最大值 | `asc_duplicate_scalar` + `asc_max` + `asc_reduce_max` |
| 2 | 对最大值做填充扩展 | `asc_duplicate` |
| 3 | 各元素减去最大值（防溢出） | `asc_loadalign` + `asc_sub` |
| 4 | 指数运算 | `asc_exp` |
| 5 | 每行求和 | `asc_duplicate_scalar` + `asc_add` + `asc_reduce_sum` |
| 6 | 对求和结果做填充扩展 | `asc_duplicate` |
| 7 | 归一化并写回 | `asc_div` + `asc_storealign` |

### 两个容易踩的坑

1. **单条向量指令有宽度上限。** 一行有 `WIDTH = 128` 个 `float`，但一条向量指令一次处理不了这么多，需要拆成 `repeat_times` 次。每次能处理的元素个数 `one_repeat_cnt` 由核函数传入（**不要在 `softmax_vf()` 内调用 `asc_get_vf_len()`**——它是 `__aicore__` 接口，`__simd_vf__` 函数内调用会编译报错 `simd_vf function can only call simd_callee function`）。拆分后尾部的掩码用 `asc_update_mask_b32(count)` 递减处理。
2. **写 `exp_ub` 与读回 `exp_ub` 之间必须同步。** 第一部分的 `Exp` 结果要写回 UB、第二部分再读回来做 `ReduceSum`，中间需要用 `asc_mem_bar(VST_VLD)` 保证写操作对后续读可见。

如果确实需要把 VF 内的逻辑拆成辅助函数，辅助函数**必须**用 `__simd_callee__` 限定，否则同样会报上面的编译错误。

## 环境准备

| 类别 | 需要的组件 |
| :--- | :--- |
| 硬件与系统 | Ascend 950PR / Ascend 950DT 系列 NPU（NPU 架构 `dav-3510`）、配套驱动和固件 |
| CANN | CANN 9.1.0 或更高版本 |
| 构建工具 | CANN 自带的 ASC 编译器（bisheng）、CMake ≥ 3.16 |
| 离线仿真（可选） | CANN 自带的 NPU Simulator（`cannsim` / `npusim`），无真实设备时可离线运行 |

本题涉及的 RegBase 接口（`c_api/asc_simd.h`、`__simd_vf__`）仅在 **Ascend 950PR / Ascend 950DT**（NPU 架构 `dav-3510`）上支持，在其他芯片上编译或运行会失败。接口头文件可直接查阅环境中的 `$ASCEND_TOOLKIT_HOME/x86_64-linux/asc/include/c_api/`。

先加载 CANN 环境；如果 CANN 不在示例路径，请改成实际的 `set_env.sh` 路径：

```bash
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
```

## 运行验证

在复制出的参与者目录下运行：

```bash
# 注意：在线环境一般不用修改脚本，可根据实际情况修改脚本中激活 CANN 环境的路径
bash run.sh
```

`run.sh` 会自动执行 `cmake .. && make -j && ./softmax_custom`。通过时输出：

```
Output: 0.010085769 0.011691351 0.005982758 0.0051221349 0.0047386498 0.0081835389 0.0070658154 0.0054982905 ...
Golden: 0.010085773 0.011691355 0.0059827599 0.0051221363 0.0047386512 0.0081835417 0.0070658177 0.0054982919 ...
Max abs err: 9.3132257e-09
[Success] Case accuracy verification passed.
```

host 侧按与 kernel 相同的公式（`max → exp(x-max) → sum → div`）逐行计算 golden，并用**最大绝对误差 `abs_err < 1e-4`** 做容差比对。骨架未补全时 `softmax_vf()` 为空函数，`dst_ub` 不会被写入，程序能编译通过但会输出 `[Failed] ...`，这是正常的起点状态。

**没有设备也能做题**：可用 CANN 自带的 NPU Simulator 做离线仿真运行（旧版本命令名为 `cannsim`，2026 年 7 月 30 日之后的版本更名为 `npusim`），它集成在 CANN toolkit 包中，**不依赖驱动和固件**：

```bash
# 1. 先编译出可执行文件（编译只需要 CANN，不需要真实设备）
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build && cd build && cmake .. && make -j

# 2. 用仿真器运行同一个可执行文件
cannsim record ./softmax_custom -s Ascend950 -o ./output
# 新版 CANN 中命令为：npusim record ./softmax_custom -s Ascend950 -o ./output
```

> 注意：直接 `bash run.sh` 会在编译完成后**立即执行**该可执行文件，在没有真实设备时会卡在最后一步并报错——编译产物本身已经生成。请按上面两步分开执行。仿真结果会打印在终端，同时写入 `./output/cannsim_*/cannsim.log`，看到 `[Success] Case accuracy verification passed.` 即通过。仿真器与板上运行**保持二进制兼容**，两种方式验证的是同一份代码，无需改动 `CMakeLists.txt` 或源码。
>
> 仿真约束：仅支持 **Ascend 950PR / Ascend 950DT** 芯片、仅支持**单卡**（代码中只能使用 **0 卡**）、仅支持 AI Core 计算类算子、**不支持 arm 环境**，建议运行环境为 16 核 CPU + 32GB 以上内存。

## 提交说明

将本人 `2026/CANN-Code-Detective/CANN-Camps/20261015/<你的用户名>/SoftmaxCustom/` 下补全的代码提交为 PR：

- 提交文件只需要 `softmax_custom.asc`、`CMakeLists.txt`、`run.sh` 三个文件，不要提交 `build/` 等编译产物；
- 保留 `Template/` 目录，**请勿直接修改**；
- 在 PR 描述中提供精度验证通过的截图，代码风格规范、注释清晰。

提交和 CLA 操作可参考[代码侦探活动说明](../../README.md)与 [GitCode 讨论 #287](https://gitcode.com/org/cann/discussions/287)。本题考点是 reg 矢量编程的寄存器级指令编排，建议先想清楚每一步在寄存器上怎么摆放再动手；可以合理使用 AI 辅助编程，但请确保理解每一行代码的含义。

## 推荐学习资料

- [第 4 章 Softmax 算子实践](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_V2/04_simple_operator_practice/04.02_softmax.ipynb)：本题的直接来源，完整讲解了基于指针的 C 语言编程（RegBase）下 Softmax 从接口分析到基础版实现的全过程，包括 VF 函数的 7 个步骤与 host 侧 golden / 校验写法。
- [基于指针的 C 语言编程](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_V2/03_programming_model/03.03.04_simd_c_programming_base_on_pointer.ipynb)：RegBase 的编程模型基础，讲清 `__global__ __vector__` 核函数、UB 内存申请、pipe 之间的 `asc_lock` / `asc_unlock` 同步。
- [混合编程与 VF 函数](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_V2/03_programming_model/03.05.02_hybrid_kernel_and_vf_function.ipynb)：核函数与 `__simd_vf__` 向量函数的分工关系。
- 接口头文件（最权威、随环境自带）：`$ASCEND_TOOLKIT_HOME/x86_64-linux/asc/include/c_api/`，其中 `reg_compute/reg_vector.h`（计算/归约/掩码）、`reg_load.h` / `reg_store.h`（寄存器搬运）、`vector_datamove/`（UB↔GM 搬运）、`sys_var/sys_var.h`（`asc_get_vf_len`）。

完成学习后，再回到本题目尝试独立实现。
