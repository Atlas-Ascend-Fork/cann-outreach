# 个人开发者

## 题目

代码侦探 Challenge 10 · Exp（逐元素自然指数 y[i]=e^(x[i])） 算子（Ascend C 开源仓规范化开发）

## 团队信息

- 提交者: skytosky
- 身份: 企业员工
- 单位: 个人开发者

## 成员

- skytosky (skytosky): 提交者

## 实现说明

- Tiling（op_host/exp_tiling.cpp）：查询平台信息（AIV 核数 / UB 大小）→ shape 逐维相乘得 totalNum
  → 数据量小于 2048/核 时缩减启动核数 → blockFactor 按 16 元素对齐（32B 对齐搬运）均分
  → ubFactor 按 UB 容量封顶 2048 → 按 dtype 设置 tilingKey（FP16→MODE_0，FP32→MODE_1）
  → 空 tensor 防护（totalNum==0 时下发安全值直接返回）
- Kernel（op_kernel/exp.h / exp.cpp）：Vector 三段式流水（CopyIn → Compute → CopyOut），
  双缓冲（BUFFER_NUM=2）；按 blockFactor 多核切分（末核吸收余量），单核内按 ubFactor 再分块，
  尾块按实际剩余元素数处理；计算核心为 AscendC::Exp 逐元素指令
- schMode 模板参数：0 → half 实例，1 → float 实例，核函数入口按 if constexpr 分发
