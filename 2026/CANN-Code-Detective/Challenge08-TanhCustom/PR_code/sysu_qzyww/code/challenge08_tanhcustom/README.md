# challenge08_tanhcustom

提交团队: 中山大学
提交者: wenjingwen


## 算子实现介绍

基于 Ascend C 在 ascend910b 上实现逐元素 Tanh 激活算子，支持
  float32/float16（tilingKey 按 dtype 自动选择模板实例）。

  Tiling 策略：输入 (8, 2048) 共 16384 个元素，SetBlockDim(8) 切到 8 核并行，每核
  blockLength=2048；核内按 tileNum=16 分块、双缓冲（BUFFER_NUM=2），单块
  tileLength=64，CopyIn/Compute/CopyOut 三级流水掩盖搬运时延。

  计算公式：采用数值稳定形式 y = tanh(x) = 1 − 2/(e^{2x}+1)，通过 Exp → Mul → Adds
  → Duplicate → Div → Adds 六条向量指令完成。相比直接实现
  (e^x−e^−x)/(e^x+e^−x)，该形式在 |x| 较大时不会出现 inf/inf=NaN（fp32 下 |x|&gt;88
  即触发）；用高精度 Div 替代近似 Reciprocal（vrec 查表精度仅约 2⁻¹⁰）。

  验证结果：上板实测 float32 全量 maxAbsErr=2e-7、float16
  maxAbsErr=1.6e-3（均为对应 dtype 舍入级别），含 ±100、0、1e-6
  等边界特征点，多次重复运行结果稳定。
