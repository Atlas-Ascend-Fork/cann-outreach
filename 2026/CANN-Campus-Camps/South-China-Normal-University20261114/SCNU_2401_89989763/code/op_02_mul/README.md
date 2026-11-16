# op_02_mul

提交团队: 华南师范大学
提交者: 杨靖杰


## 算子实现介绍

算子实现：z = x * y，支持 float32 / float16，数据格式 ND，输入输出形状均为 (8, 2048)，元素总数 16384。

1) Host 侧（op_host/mul.cpp）
   通过 PlatformAscendC 获取 AIV 核数 num_cores_aiv 和 UB 容量；从 GetRequiredInputTensor(0)
   取 x 的 dtype、GetShapeSize() 得到 length_x = 16384，写入 MulTilingData.length。
   tiling key 由输入 dtype 通过 ASCENDC_TPL_SEL_PARAM(context, DT_X) 选择，交由
   tiling_key_mul.h 中的 ASCENDC_TPL_DATATYPE_DECL 分派到 float / half 两个实例，
   避免 kernel 侧运行时判断 dtype。SetBlockDim(num_cores_aiv) 启动全部 AIV 核，
   workspace 配置为 0（无额外临时空间需求）。

2) 核间切分
   KernelMul&lt;DT_X&gt; 为模板类，DT_X 取 float 或 half。每核长度按
   perCore = ceil(totalLength / blockNum) 计算后向上对齐到 32 字节
   （fp32 为 8 元素、fp16 为 16 元素的整数倍），本核起始偏移 start = blockIdx × perCore。
   若 start ≥ totalLength，说明该核无数据，blockLength / tileLength / tileNum 置 0 并直接返回，
   不发起零长度 DataCopy；否则 blockLength 取 perCore 与 totalLength - start 的较小值。
   之所以先向上取整再截断，是因为 16384 不能被常见 AIV 核数整除，直接整除会丢失元素。

3) 核内 UB 分块
   单次搬运 tileLength = 8192 / sizeof(DT_X)（fp32 为 2048 元素、fp16 为 4096 元素），
   超过本核长度时收缩为 blockLength；分片数 tileNum = ceil(blockLength / tileLength)。
   X / Y / Z 三条队列各开 2 块 buffer，每块 tileLength × sizeof(DT_X) = 8KB，
   UB 峰值占用 3 × 2 × 8KB = 48KB。

4) 计算流程
   Process 按 tileNum 循环，每片执行 CopyIn → Compute → CopyOut：
   - CopyIn：DataCopy 将 x、y 从 GM 搬入 UB，AllocTensor 取 buffer，EnQue 入队；
   - Compute：DeQue 取出 x、y，AllocTensor 取 z 的 buffer，调用 AscendC 矢量指令
     Mul(zLocal, xLocal, yLocal, len) 完成逐元素乘法，EnQue 入队，并释放 x、y 的 buffer；
   - CopyOut：DeQue 取出 z，DataCopy 搬回 GM，FreeTensor 释放。
   分片的 GM 偏移与长度均为 32 字节整数倍，满足 DataCopy 的对齐要求，末片由
   blockLength - offset 收紧，天然对齐，无需 DataCopyPad。

5) 同步方式
   全程依赖 TQue 的 AllocTensor / EnQue / DeQue / FreeTensor，在 MTE2（搬入）、
   V（计算）、MTE3（搬出）之间自动插入 SetFlag / WaitFlag，未手写同步指令。

6) 精度说明
   全程使用输入 dtype 直接参与计算，不做额外的类型提升或中间转换。fp32 结果与 golden 一致；
   fp16 在输入取值 [1, 10]、乘积不超过 100 的量级下 ulp 约 0.06，与以 float64 计算后回写
   float16 的 golden 相比差异不超过 1~2 ulp，处于验证集容差范围内。
