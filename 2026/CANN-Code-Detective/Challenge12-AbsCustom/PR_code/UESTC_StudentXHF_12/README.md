# Challenge12 AbsCustom — StudentXHF

基于官方 `Template/code_temple` 完成逐元素绝对值算子，支持题目指定的 ND `(8, 2048)`、float32 和 float16 输入。Host tiling 将 16384 个元素均分至 8 个 AIV 核，每核 2048 个元素，以 256 元素为一块；kernel 通过 `TPipe` 和双缓冲队列依次执行 `DataCopy`、`AscendC::Abs`、`DataCopy`。

验证：StudentXHF 的 [CANNJudge 提交 1206404](https://cannjudge.cn/cann_code_detective/challenge12-abscustom/challenge12-abscustom/submission/6aeecac66bd0f9be374cf7c1) 为 **PASS 2/2**，两个测试点输出错误占比均为 0.00%。仓库中同步更新 host tiling UT 期望值，并将 kernel UT 输入改为包含负数、零和正数的确定性数据。

活动：[第十二期 Issue #935](https://gitcode.com/cann/cann-learning-hub/issues/935)。
