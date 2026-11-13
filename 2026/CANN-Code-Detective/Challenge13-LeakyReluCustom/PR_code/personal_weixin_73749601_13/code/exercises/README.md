# 算子工程调测

加载 CANN 9.0.0 或 9.1.0 环境，安装 Python 参考实现需要的 numpy、ml_dtypes（以及参考文件实际使用的 torch/tensorflow）。

```bash
source /path/to/cann/set_env.sh
bash build.sh -j4       # 构建 RUN 包
DEVICE_ID=0 bash build.sh -j4 -e  # 构建并运行首个测试点、比较输出
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include GTEST_SOURCE_DIR=/path/to/googletest bash build.sh -j4 -u # 本地 UT
```

GoogleTest 使用本地源码（建议1.14.0），nlohmann/json 头文件建议3.11.3（include目录下应有nlohmann/json.hpp），不自动下载。CPU 调测还需要工具链的 tikicpulib。修改 op_host、op_kernel，以及存在时的 op_api。kernel为空时输出校验失败是正常现象。samples 只包含第一个 NPU 测试点。输入按题目规范生成，比较前检查文件大小。

opapi 工程的完整结果由外层 NPU example 校验；内层 CPU UT 的 fixture 需要按实际内外层转换关系自行提供，不能直接套外层 GT。默认 -u 明确跳过内层数值断言。准备并编辑内层 fixture 后可使用 `ENABLE_INNER_KERNEL_UT=ON bash build.sh -u` 启用；其结果需按内层语义自行断言。API UT 需要设备上下文，默认明确跳过；先构建 RUN，再使用 `RUN_OP_API_UT=1 DEVICE_ID=0 bash build.sh -u` 执行外层 workspace 接口测试，该测试不代替 -e 的数值校验。若修改 tiling 结构，也需要同步调整 kernel UT 的 tiling fixture。

# 算子实现介绍

提交团队: 个人
提交者: Cola_Coder

系列首个带属性算子：op_def 注册 negative_slope 属性，Tiling 经 GetAttrs()->GetAttrPointer<float>(0) 读取并写入 tilingData.negativeSlope 下发 kernel。多核切分：tiling 按 MIN_SPLIT_THRESHOLD=1024 控制每核最少数据量，blockFactor = CeilDiv(totalNum, usedCoreNum)，ubFactor 按双队列×双缓冲+zero/neg 两个临时 Tensor 并预留一半 UB 给高阶 API 临时空间预算。kernel 采用 CopyIn→Compute→CopyOut 三段式流水，TBuf<VECCALC> 分配 zeroBuf/negBuf，Duplicate 广播 0 与 α 标量、Max/Min 取正负半轴、Mul/Add 合成 y = max(x,0) + α·min(x,0)；zeroBuf 复用为 α 向量节省 UB。实际分块 cap 到 4096 元素并向上对齐到 32B，对齐走 DataCopy 快路径，非对齐尾部回退 DataCopyPad。CANNJudge 验证集 PASS（precision_ratio=1）。
