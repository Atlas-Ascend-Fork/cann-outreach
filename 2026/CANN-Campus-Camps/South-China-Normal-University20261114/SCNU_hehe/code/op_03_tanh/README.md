# op_03_tanh

提交团队: 华南师范大学
提交者: hehe_113


## 算子实现介绍

一、算子功能
    实现逐元素双曲正切 y = tanh(x) = (exp(x) - exp(-x)) /
    (exp(x) + exp(-x))。输入 x 与输出 y 形状均为 (8, 2048)，
    元素总数 16384，支持 float32 / float16，数据格式 ND。
    约束：不得直接调用 tanh 高阶接口，必须用小算子拼接实现。

二、开发环境与提交形态
    平台：CANNJudge（华南师范大学专场 20261114，Lesson 3），
    CANN 版本 9.0.0，目标芯片 ascend910b。
    工程为开源仓规范化算子工程，共 38 个文件，构建脚本
    build.sh 生成 RUN 包，examples 再以 aclnn 接口跑数值对比。
    可编辑文件 7 个，实际只需改两个：
        op_host/tanh_tiling.cpp  —— 补 Tiling 实现
        op_kernel/tanh.h         —— 补计算过程与临时变量
    op_def / infershape / tiling_data / tiling_key / 入口函数
    均已由模板写好，本次未改动。

三、实现思路
    1. 计算域统一为 float。fp32 输入直接参与运算；fp16 输入先用
       Cast 提升到 float，算完再一次 Cast 降回 half。这样 fp16 的
       结果等价于「fp32 精度 + 一次舍入」，与参考实现的差异仅剩
       末尾舍入。
    2. 公式按题目给定形式展开，不化简为 1 - 2/(exp(2x)+1)。
       后者在 x = 10 时需计算 exp(20) = 4.85e8，超出 fp16 上限
       65504 会直接溢出为 inf；给定形式只需 exp(±10) = 22026 与
       4.54e-5，两种精度都能容纳。
    3. 多核切分沿用「向上取整 + 32B 对齐」：核数由硬件决定，
       16384 未必除得尽，每核长度取 ceil 后按对齐元素数上取整，
       余量由最后一核吸收。
    4. 核内按 ubFactor 分块循环，本题每核仅几百个元素，
       循环次数恒为 1，省去多余的循环与同步开销。
    5. 双缓冲：搬入、搬出各 2 块 UB，使 MTE2 / V / MTE3
       三条流水可重叠。

四、关键实现
    Tiling 侧（tanh_tiling.cpp）
        totalNum   ：由输入 shape 的 GetShapeSize 取得。
        typeSize   ：fp16 / bf16 取 2，其余取 4，手动判定，
                     不依赖额外的类型工具头文件。
        alignNum   ：32 / typeSize，即 32B 对应的元素个数，
                     fp32 为 8，fp16 为 16。
        blockFactor：CeilDiv(totalNum, coreNum) 后再 CeilAlign，
                     保证每个核的起始地址都是 32B 整数倍。
        usedCoreNum：CeilDiv(totalNum, blockFactor)，数据量小时
                     不必把全部核叫醒。
        ubFactor   ：按每元素 4×typeSize + 5×4 字节的预算推导，
                     减去 32KB 框架余量后 FloorAlign，再夹到
                     blockFactor 以内。
        tilingKey  ：fp16 走 schMode 0，fp32 走 schMode 1。
    Kernel 侧（tanh.h）
        Init    ：算本核起始下标与长度，为空核直接返回；按偏移
                  设置两个 GlobalTensor；为两条队列与 5 块
                  float 缓存申请 UB，长度统一按 32B 向上取整。
        CopyIn  ：DataCopy 把 x 的一块从 GM 搬入 UB 并入队。
        Compute ：取数后按 dtype 分两条路径 —— fp32 直接进
                  TanhCore；fp16 先 Cast 成 float，算完再 Cast
                  回 half。最后入队并释放输入张量。
        CopyOut ：从输出队列取结果，DataCopy 写回 GM 并释放。
        Process ：按 ubFactor 分块循环执行上述三段。
        TanhCore：核心六条指令，全部为最小算子
                  1) Exp(expPos, x)          exp(x)
                  2) Muls(tmp, x, -1)        求 -x
                  3) Exp(expNeg, tmp)        exp(-x)
                  4) Add(sum, expPos, expNeg) 分母
                  5) Sub(tmp, expPos, expNeg) 分子
                  6) Div(dst, tmp, sum)      相除
                  中间量分别落在 5 块 float 缓存上，任何一条指令的
                  目的张量都不与源张量重叠，不存在原地写回。

五、正确性要点
    1. 精度路线不能想当然。实测 fp32 走纯公式最大误差 1.19e-7，
       在 1e-4 容差下 0 个元素超差；而 fp16 若全程用 half 计算，
       误差可达 9.8e-4，仅有 1e-3 以上容差才能通过。因此 fp16
       必须走 float 中转，这是本题通过与否的关键。
    2. 禁止使用 tanh 接口。实现中只出现 Exp / Muls / Add / Sub /
       Div 五个基础接口，未调用任何高阶激活算子。
    3. 减法与除法的操作数顺序：Sub 与 Div 均为 src0 作用于 src1，
       分子分母写反会得到 1/tanh(x)，属于静默错误。
    4. AllocTensor 与 FreeTensor 必须配对：输入在 Compute 内释放，
       输出在 CopyOut 内释放，否则循环中队列被占满而阻塞。
    5. 起点必须对齐：blockFactor 按 alignNum 上取整，保证每个核
       的 GlobalTensor 起始地址与单块搬运字节数都是 32B 倍数。
    6. 空核与空张量保护：起始下标越过总长度时长度置 0 直接返回；
       totalNum 为 0 时 Tiling 提前返回，避免 CeilDiv 除零。

六、验证结果
    提交 ID 1344235，状态 Pass，2/2 测试点通过，
    输出错误占比均为 0.00%，用时 15.06 us 与 15.41 us，
    当前最优记录为 4.47 us 与 4.54 us。
    提交前另做离线校验：核数取 8/20/24/40/48/64/80 与 fp32、
    fp16 两种 dtype 共 14 种组合，每个元素覆盖次数恰为 1，
    无空洞、无重复，所有搬运均满足 32B 对齐；
    单核 UB 占用最高为 8 核 fp32 场景的 73728 字节，约占 UB
    总量的 28%，余量充足。

七、性能分析
    每核仅 3 条 DMA 指令（搬入 x、搬出 y 各一条，加一次队列
    调度），单次搬运约 1.6 KB；核内则固定 6 条矢量指令外加
    fp16 场景的 2 条 Cast。指令数已压到与数据量匹配的最小规模。
    与最优记录的约 3 倍差距主要来自 aclnn 派发链路的固定开销：
    算子注册、tiling 解析、任务下发等，属框架层成本，非核内
    计算所致。继续优化内核收益有限。

八、可扩展性
    1. 归一化指数是同类激活算子的公共骨架。把 TanhCore 末尾的
       Div 换成别的收尾即可得到 Sigmoid：1/(1+exp(-x))，
       前两步 Muls + Exp 可直接复用。
    2. 增加新 dtype（如 bf16）只需三处同步：op_def 的 DataType
       列表、tiling_key 的 C_DT 枚举、Tiling 中 typeSize 的判定；
       kernel 侧 sizeof(T) 分支会自动走 float 中转路径。
    3. 若要支持任意长度且不要求 32B 整除，末块需用 DataCopyPad
       搬运并在计算侧按实际长度裁剪。
