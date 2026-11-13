# op_04_relu

提交团队: 华南师范大学
提交者: 李俊杰


## 算子实现介绍

Host 侧：TilingFunc 从输入 shape 连乘得 totalNum，通过 PlatformAscendC 取 AIV 核数
与 UB 大小，按 32B 对齐规则计算 blockFactor（每核元素数）与 ubFactor（每块元素数），
SetBlockDim(coreNum) 配置启动核数，并按 dtype 用 GET_TPL_TILING_KEY 选择
half / float 模板实例。
Kernel 侧：每核按 blockIdx×blockFactor 定位 GM 起点，尾核按 totalNum 裁剪；
按 ubFactor 分块循环，走 CopyIn→Compute→CopyOut 三段式流水，
队列各分配 2 块 UB 形成双缓冲，搬运长度均为 32B 整数倍；
计算为 y = max(0, x)，单条 AscendC::Relu 完成。
