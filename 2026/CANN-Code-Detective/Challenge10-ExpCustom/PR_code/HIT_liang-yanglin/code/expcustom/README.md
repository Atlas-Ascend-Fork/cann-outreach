# expcustom

提交团队: 哈尔滨工业大学
提交者: 梁杨琳


## 算子实现介绍

一、算子功能
基于 Ascend C 在昇腾 NPU 上实现自然指数算子，逐元素计算 y = e^x。
输入 x：float32/float16，ND 格式，形状 (8, 2048)；输出 y 与 x 同 dtype 同形状。

二、实现结构（开源仓规范化）
op_host：exp_def.cpp 算子注册、exp_infershape.cpp 形状推导（输出同输入 shape）、exp_tiling.cpp Tiling 函数。
op_kernel：exp.cpp 核函数入口、exp.h 算子类（CopyIn-Compute-CopyOut 三段式）、exp_tiling_data.h / exp_tiling_key.h 数据与模板参数定义。

三、Tiling 实现（exp_tiling.cpp）
读取输入 dtype，通过 ASCENDC_TPL_SEL_PARAM 按模板参数自动选择 tilingKey（fp32 选 float 实例，fp16 选 half 实例）；
输入 shape 逐维相乘得到总元素数 totalLength = 16384；设置 tileNum = 1；
SetBlockDim(8) 启动 8 核并行；workspace 置 0。

四、核函数实现（exp.h / exp.cpp）
多核切分：blockLength = totalLength / 8 = 2048，每核按 GetBlockIdx() 偏移设置 GM 起始地址，整除无尾块；
InitBuffer 为输入/输出队列分配 UB；每核一次 DataCopy 搬入完整分块（fp32 8KB / fp16 4KB，32B 对齐），
调用 AscendC::Exp 逐元素计算，再一次 DataCopy 搬回 GM。
性能策略：整块单搬（tileNum=1）+ 单缓冲（BUFFER_NUM=1）。实测 910B4 上 kernel 耗时由 7.86us 降至 2.23us（约 3.5 倍）。

五、验证结果
UT：op_host Tiling 单测通过（tilingKey/tilingData 断言正确），op_kernel CPU 模拟通过，与 np.exp 比对 MERE/MARE = 0；
真机：aclnnExp 调用正常，exp(1.0) 输出最大误差 8.3e-8（fp32 ulp 级）；
CANNJudge 平台验证两测试点均 PASS，错误占比 0.00%。
