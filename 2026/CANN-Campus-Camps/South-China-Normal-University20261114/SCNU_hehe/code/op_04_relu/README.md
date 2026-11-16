# op_04_relu

提交团队: 华南师范大学
提交者: hehe_113


## 算子实现介绍

一、算子功能
    实现逐元素修正线性单元 y = max(0, x)，对应 PyTorch 的
    torch.nn.functional.relu。输入 x 与输出 y 形状均为 (8, 2048)，
    元素总数 16384，支持 float32 / float16，数据格式 ND。
    本题未限制必须用基础接口拼接，因此直接以单条矢量指令完成。

二、开发环境与提交形态
    平台：CANNJudge（华南师范大学专场 20261114），CANN 9.0.0，
    目标芯片 ascend910b。工程为开源仓规范化算子工程，共 34 个
    文件，构建脚本 build.sh 生成 RUN 包，examples 再以 aclnn 接口
    跑数值对比。可编辑文件 7 个，实际只需改两个：
        op_host/relu_tiling.cpp  —— 补 Tiling 实现
        op_kernel/relu.h         —— 补五个函数体
    op_def、infershape、tiling_data、tiling_key、kernel 入口以及
    relu.cpp 中的 dtype 分发均由模板写好 。
三、实现思路
    1. 不需要 float 计算域。ReLU 只做比较与截断，不涉及超越
       函数，fp16 直接在本类型上计算即为精确结果，不像 Tanh
       那样必须用 Cast 中转来保住精度。
    2. 计算只需一条指令：Maxs 求「张量元素与标量的较大者」，
       标量取 0 即为 relu，比用 Max 配一块全零张量省一次填充。
    3. 多核切分沿用「向上取整 + 32B 对齐」：核数由硬件决定，
       16384 未必除得尽，每核长度取 ceil 后按对齐元素数上取整，
       余量由最后一核吸收。
    4. 核内按 ubFactor 分块循环，本题每核仅几百个元素，循环
       次数恒为 1。
    5. 双缓冲：搬入、搬出各 2 块 UB，使 MTE2 / V / MTE3 三条
       流水可重叠。

四、关键实现
    Tiling 侧（relu_tiling.cpp）
        totalNum   ：由输入 shape 的 GetShapeSize 取得。
        typeSize   ：fp16 / bf16 取 2，其余取 4，手动判定。
        alignNum   ：32 / typeSize，fp32 为 8，fp16 为 16。
        blockFactor：CeilDiv(totalNum, coreNum) 后再 CeilAlign。
        usedCoreNum：CeilDiv(totalNum, blockFactor)，数据量小时
                     不必把全部核叫醒。
        ubFactor   ：按每元素 4×typeSize 字节的预算推导（只有
                     in / out 两队列、各 2 块 buffer），减去 32KB
                     框架余量后 FloorAlign，再夹到 blockFactor 内。
        tilingKey  ：fp16 走 schMode 0，fp32 走 schMode 1。
    Kernel 侧（relu.h）
        Init    ：算本核起始下标与长度，为空核直接返回；按偏移
                  设置两个 GlobalTensor；为两条队列申请 UB。单次
                  长度取 min(ubFactor, 本核长度, 8192) 三重夹紧，
                  保证 InitBuffer 永不越界。
        CopyIn  ：DataCopy 把 x 的一块从 GM 搬入 UB 并入队。
        Compute ：取数后一条 Maxs 完成，入队并释放输入张量。
        CopyOut ：从输出队列取结果，DataCopy 写回 GM 并释放。
        Process ：按块长循环执行上述三段，末块取剩余长度。

五、正确性要点
    1. 等价写法有陷阱。用 Sub / Muls 组合实现 relu 需先求 |x|
       再折半相加，既多指令又引入符号判定；直接用 Maxs 求与 0
       的较大者，与原语义完全一致。
    2. AllocTensor 与 FreeTensor 必须配对：输入在 Compute 内
       释放，输出在 CopyOut 内释放，否则循环中队列被占满而阻塞。
    3. 起点必须对齐：blockFactor 按 alignNum 上取整，保证每个核
       的 GlobalTensor 起始地址与单块搬运字节数都是 32B 倍数。
    4. 空核与空张量保护：起始下标越过总长度时长度置 0，Process
       直接返回；totalNum 为 0 时 Tiling 提前返回，避免除零。
    5. UB 上限不能全靠 host 保证。工程自带的 kernel 单测把
       ubFactor 直接写成 92160，fp32 下即 368KB，远超 256KB 的
       UB 容量。因此核内对块长做 8192 元素硬上限，host 给多大
       都能降级循环跑完。
    6. tilingKey 与 dtype 必须一致：fp16 走 schMode 0 实例化
       half，fp32 走 schMode 1 实例化 float，写反会按错误精度
       解释同一片内存。

六、验证结果
    提交 ID 1346284，状态 Pass，2/2 测试点通过，输出错误占比
    均为 0.00%，用时 16.52 us 与 27.46 us，当前最优记录为
    3.81 us 与 15.20 us。
    提交前另做离线校验：核数取 1/8/20/24/32/40/48/64/80 与
    fp32、fp16 两种 dtype 共 18 种组合，每个元素覆盖次数恰为 1，
    无空洞、无重复，所有搬运均满足 32B 对齐。单核 UB 占用最高
    出现在单核 fp32 场景，131072 字节约占 50%；常规 40 核场景
    仅占 2.5%。

七、性能分析
    以 40 核切分为例，每核 416 个元素、单块 1.6KB，只需 3 条
    DMA 指令与 1 条矢量指令，指令数已压到理论下限。
    小算例用时约为最优记录的 4.3 倍，大算例回落到约 1.8 倍。
    倍数随数据量增大而收敛，说明差距主要来自 aclnn 派发链路的
    固定开销：算子注册、tiling 解析、任务下发，属框架层成本，
    并非核内计算所致。
