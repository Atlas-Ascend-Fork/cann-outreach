# 赛诺信致（北京）软件技术有限公司

## 团队信息

- 提交者: 小东
- 身份: 企业员工
- 单位: 赛诺信致（北京）软件技术有限公司

## 成员

- 小东 (ciknife): 提交者

## 算子: challenge07_sigmoidcustom

多核实现，UB 双缓冲流水线。Tiling 策略：总元素数按 AIV 核数均分，blockFactor 向上对齐到 16 元素（32B，兼容 float32/float16）；数据量小于 1024×核数 时自动减少启动核数；ubFactor 按 UB 容量自适应（输入/输出 2 队列 × BUFFER_NUM=2），尾核取剩余元素防越界。Kernel 侧 CopyIn → AscendC::Sigmoid → CopyOut 三段式，逐元素计算 y = 1 / (1 + exp(-x))，最后一个 tile 取剩余量。
