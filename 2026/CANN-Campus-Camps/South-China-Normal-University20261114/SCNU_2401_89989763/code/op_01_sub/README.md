# op_01_sub

提交团队: 华南师范大学
提交者: 杨靖杰


## 算子实现介绍

算子实现：z = x - y，支持 float32 / float16，数据格式 ND。

1) Host 侧（run_kernel）：从 TensorGroupInfo 中第一个 TensorInfo 的 shape 逐维相乘得到元素总数
   totalLength = 8 × 2048 = 16384；读取 dtype 字段（0=fp32，1=fp16）决定走哪个核函数实例
   （LocalTensor 的 dtype 是编译期确定的，故用模板类 KernelSub&lt;float&gt; / KernelSub&lt;half&gt;
   包装成两个 __global__ __vector__ 入口）。blockNum 取 min(availableCoreNum,
   totalLength / 每 32B 对齐块元素数)，避免可用核数多于数据块数而产生空搬运。

2) Tiling 策略：多核均分 + 32B 对齐。perBlock = ceil(totalLength / blockNum) 向上对齐到
   32 字节（fp32 为 8 元素、fp16 为 16 元素的整数倍），本核起始偏移 start = blockIdx × perBlock，
   长度按剩余元素数截断；start ≥ totalLength 的尾核直接返回，不发起零长度 DataCopy。

3) UB 切分：单次搬运 tileLen 取 8KB / sizeof(T)（fp32 2048 元素、fp16 4096 元素），
   tileLen 超过本核长度时收缩为本核长度；用 while 循环把本核负责区间切成若干片，
   逐片执行 CopyIn → Compute → CopyOut。X / Y / Z 三条队列各开 1 块 buffer，
   UB 峰值占用约 3 × 8KB = 24KB。

4) 计算：调用 AscendC 矢量指令 Sub(dstLocal, xLocal, yLocal, len) 完成逐元素减法；
   fp32 / fp16 同构，无需类型提升或中间转换。

5) 同步：通过 TQue 的 AllocTensor / EnQue / DeQue / FreeTensor 在 MTE2、V、MTE3 之间
   自动插入 SetFlag/WaitFlag，保证搬运与计算流水正确，无需手写同步。

6) 精度：全程按输入 dtype 直接相减，未做额外精度转换；GM 的读写偏移均按 32B 对齐，
   结果与以 float64 计算后回写原 dtype 的 golden 完全一致（rtol=1e-4、atol=0.1）。
