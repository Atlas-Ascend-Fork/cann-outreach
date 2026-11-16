# op_02_mul

提交团队: 华南师范大学
提交者: hehe_113


## 算子实现介绍

一、算子功能
    实现逐元素矢量乘法 z = x * y。输入 x、y 与输出 z 形状均为
    (8, 2048)，元素总数 16384，支持 float32 / float16，格式 ND。

二、开发环境与提交形态
    平台：CANNJudge（CANN 训练营-华南师范大学专场 20261114，
    Lesson 2），CANN 版本 9.0.0，目标芯片 ascend910b。
    与 Lesson 1 的单文件直调不同，本题是标准自定义算子工程：
    op_host 负责 tiling 与形状、类型推导，op_kernel 负责核函数，
    由 npu_op_code_gen 自动生成 aclnn 接口。
    实际改动落在 op_host/mul.cpp 与 op_kernel/mul.cpp 两个文件。

三、实现思路
    1. 核数由硬件决定：host 侧调用 SetBlockDim(GetCoreNumAiv())，
       把全部 AIV 核拉起，因此 16384 未必能被核数整除。
    2. 向上取整加对齐的切分：每核长度先取 ceil(16384 / blockNum)，
       再按 32 字节对齐所需元素数上取整（fp32 为 8，fp16 为 16）；
       核 i 处理区间 [i * perCore, min((i+1) * perCore, length))。
       切分余量由最后一核吸收，既不漏算也不重算。
    3. 核内按 TILE_LENGTH = 1024 分块循环。每核通常只有几百个元素，
       循环次数退化为 1 次，省掉多余的循环与同步开销。
    4. 双缓冲：三个队列各申请 2 块 UB 内存，使搬运与计算重叠，
       用以掩盖 DMA 延迟。
    5. 类型泛化：KernelMul&lt;DT_X&gt; 由 tiling key 选择 float 或 half
       实例，sizeof(DT_X) 自动决定对齐粒度与 UB 申请量。

四、关键实现
    TilingFunc   ：读取平台 AIV 核数、输入 dtype 与元素总数，写入
                   MulTilingData.length，设置 blockDim 与 tiling key，
                   workspace 置 0（本算子无需额外空间）。
    InferShape   ：z 的形状直接复制 x 的形状。模板中该函数原本为空，
                   必须补全，否则 aclnn 运行期取不到输出形状。
    InferDataType：输出 dtype 透传输入 dtype，并做空指针保护。
    Init         ：计算 perCore、本核起始下标与长度；分不到数据的核
                   将长度置 0 直接返回；按起始下标设置三个
                   GlobalTensor，并为三队列申请 UB。
    Process      ：按 1024 分块循环，逐块执行 CopyIn / Compute /
                   CopyOut，末块长度按剩余量收缩。
    Compute      ：AscendC::Mul(zLocal, xLocal, yLocal, curLen)。
                   乘法满足交换律，两个操作数顺序不影响结果。

五、正确性要点
    1. 整除假设不能沿用。本题核数由硬件决定，perCore 必须向上取整，
       沿用 Lesson 1 的向下取整会整体丢弃末段元素。
    2. 起点必须对齐：perCore 按 alignNum 上取整，保证每个核的起始
       地址是 32 字节整数倍；单块搬运字节数同样为对齐倍数。
    3. 空核保护：若某核起始下标已越过总长度，长度为 0，Process
       立即返回，避免越界访问。
    4. AllocTensor 与 FreeTensor 必须配对：两个输入在 Compute 内释放，
       输出在 CopyOut 内释放，否则循环中队列被占满而阻塞。
    5. tiling key 与 kernel 模板参数必须一致：host 侧
       ASCENDC_TPL_SEL_PARAM 传入的 dtype 决定编译期实例化哪一个。
    6. DataCopy 参数顺序：目标在前。搬入为 (Local, Global, len)，
       搬出为 (Global, Local, len)。

六、验证结果
    提交 ID 1332231，状态 Pass，2/2 测试点通过，
    输出错误占比 0.00%，用时 21.26 us 与 15.93 us，
    当前最优记录为 4.06 us 与 3.75 us。
    切分逻辑另做离线模拟：核数取 8/20/32/40/48/64 与 fp32、fp16
    两种 dtype，共 12 种组合，每个元素覆盖次数恰为 1，无空洞、
    无重复，所有搬运均满足 32 字节对齐。

七、性能分析
    每核仅 3 条 DMA 指令（搬入 x、y，搬出 z），单次约 1.6 KB，
    kernel 侧已接近最小指令数，差距不在核内计算。
    与最优记录的约 5 倍差距主要来自 aclnn 派发链路的固定开销：
    算子注册、tiling 解析、任务下发等，属框架层成本。
    进一步压缩的空间有限；若追求极限可减少核数、让单核搬更大块，
    但会牺牲并行度，收益不确定，本次未采用。

八、可扩展性
    工程骨架不变，仅替换 Compute 中的一行即可得到其他二元算子：
    Add 加法、Sub 减法、Max 取大、Min 取小。
    新增 dtype（如 bf16、int32）需同步修改三处：OpDef 的 DataType
    列表、tiling_key 的 C_DT 枚举；对齐粒度由 sizeof 自动适配。
    若要支持任意长度且不要求 32 字节整除，尾块需补零搬运，
    并在写回时按实际长度裁剪。

九、与 Lesson 1 的差异小结
    Lesson 1：单文件直调，blockDim 固定为 8，总量整齐可分，
              kernel 内假定整除，结构最简。
    Lesson 2：完整算子工程，核数随硬件变化，必须处理除不尽的情况，
              并补全 InferShape / InferDataType 两个推导函数。
    两题的 CopyIn / Compute / CopyOut 三段流水骨架完全一致，
    变化的只是切分策略与工程装配方式。
