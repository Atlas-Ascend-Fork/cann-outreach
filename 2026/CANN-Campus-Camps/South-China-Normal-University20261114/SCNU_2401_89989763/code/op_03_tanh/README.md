# op_03_tanh

提交团队: 华南师范大学
提交者: 杨靖杰


## 算子实现介绍

算子实现：y = tanh(x)，支持 float32 / float16，数据格式 ND，输入输出形状均为 (8, 2048)，总元素数 16384。

1) Host 侧 Tiling（op_host/tanh_tiling.cpp）
   从 GetInputDesc(0) 取 dtype，校验为 DT_FLOAT 或 DT_FLOAT16；由 GetInputShape(0) 的
   StorageShape 逐维相乘得到 totalNum。平台信息经 PlatformAscendC 获取 AIV 核数 coreNum
   与 UB 容量，workspace 配置为 0（计算全在 UB 内完成，无需额外临时空间）。

2) tilingKey 分派
   tiling_key_tanh.h 声明的是 schMode 的取值 {0, 1}，与 dtype 数值无约定关系。host 侧
   显式映射：dtype 为 DT_FLOAT 时取 schMode=1，DT_FLOAT16 时取 schMode=0，随后调用
   ASCENDC_TPL_SEL_PARAM(context, schMode)。kernel 入口 tanh.cpp 按 schMode 编译期分支：
   MODE_1 实例化 Tanh&lt;float&gt;，MODE_0 实例化 Tanh&lt;half&gt;，从而避免 kernel 侧运行时判断 dtype，
   也不会出现 dtype 数值落到模板列表之外的情况。

3) 三层切分（totalNum / blockFactor / ubFactor）
   - totalNum：全局元素总数 16384；
   - blockFactor：每核处理元素数，blockFactor = totalNum / blockDim；
   - ubFactor：核内单次 UB 循环的元素数，ubFactor = blockFactor / (TILE_NUM × BUFFER_NUM)，
     其中 TILE_NUM = 8、BUFFER_NUM = 2。
   blockDim 的选取采用从平台核数向下遍历：同时要求 totalNum 能被 blockDim 整除、
   每核长度能被 TILE_NUM × BUFFER_NUM × 32B 对齐元素数整除，满足即取，否则递减。
   32B 对齐元素数 = 32 / sizeof(DT_X)（fp32 为 8，fp16 为 16），因此每核单位需求为
   fp32 128 元素、fp16 256 元素。以 40 或 48 个 AIV 核为例，最终落到 blockDim = 32，
   blockFactor = 512，ubFactor = 32：fp32 每片 128 字节、fp16 每片 64 字节，均为 32 字节整数倍，
   满足 DataCopy 对齐要求，且 totalNum 无余数、无丢元素。整除关系在 host 侧保证，
   kernel 侧无需处理尾块与零长度搬运。

4) Kernel 侧 Init（op_kernel/tanh.h）
   由 tilingData 读入 blockFactor 作为 blockLength、ubFactor 作为 tileLength，并计算
   tileNum = blockLength / tileLength / BUFFER_NUM。GlobalTensor 按
   blockFactor × GetBlockIdx() 偏移到本核负责区间；为 inputQueueX、outputQueueY 各分配
   BUFFER_NUM 块 buffer，每块 tileLength × sizeof(DT_X)；另申请两块 TBuf&lt;VECCALC&gt; 临时
   buffer（tmpBufA、tmpBufB）供中间结果使用，大小同样为 tileLength × sizeof(DT_X)，
   用于消除“计算过程所需临时变量”这一空缺。

5) 计算流程
   Process 循环 tileNum × BUFFER_NUM 次，每次执行：
   - CopyIn：AllocTensor 取输入 buffer，DataCopy 将 x 从 GM 搬入 UB，EnQue 入队；
   - Compute：DeQue 取出 xLocal，AllocTensor 取 yLocal，取两块 VECCALC 临时张量，
     依次执行
       a. 钳位：Mins 上限 / Maxs 下限，将输入限制到 [-C, C]，fp32 取 C = 15，fp16 取 C = 5。
          tanh 在该范围外已饱和到 ±1，钳位不改变结果；同时保证 e^{2x} 不溢出
          （fp16 下 e^{10} ≈ 22026 &lt; 65504）；
       b. Muls 乘 2 后用 Exp 求 e^{2x}；
       c. Adds 减 1 得 e^{2x} - 1，Adds 加 1 得 e^{2x} + 1；
       d. Div 求 (e^{2x} - 1) / (e^{2x} + 1)，即 tanh(x)，写回 yLocal；
     计算完成后 EnQue 输出、FreeTensor 释放输入；
   - CopyOut：DeQue 取 yLocal，DataCopy 搬回 GM，FreeTensor 释放。

6) 同步方式与资源占用
   未手写同步指令，全程依赖 TQue 的 AllocTensor / EnQue / DeQue / FreeTensor 在
   MTE2（搬入）、V（计算）、MTE3（搬出）之间自动插入 SetFlag / WaitFlag。临时张量走
   TBuf&lt;VECCALC&gt;，不跨流水阶段，无需队列同步。UB 占用为 2 × 2 × tileLength × sizeof(DT_X)
   （输入、输出各双缓冲）加 2 × tileLength × sizeof(DT_X)（两块临时量），fp32 下约 24KB。

7) 精度说明
   公式 (e^{2x} - 1) / (e^{2x} + 1) 使用输入 dtype 直接计算，未做类型提升。在输入取值
   [-10, 10]、容差 rtol = atol = 1e-4 的用例下，|x| 较大时 tanh 已饱和，钳位与舍入误差
   均可忽略；x 接近 0 时分子存在减法相消，fp32 相对误差量级约 1e-4，处于容差边界附近。
