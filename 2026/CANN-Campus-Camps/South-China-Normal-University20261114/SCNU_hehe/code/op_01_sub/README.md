# op_01_sub

提交团队: 华南师范大学
提交者: hehe_113


## 算子实现介绍

Ascend C 矢量减法算子实现介绍

一、算子功能
    实现逐元素矢量减法 z = x - y。输入 x、y 与输出 z 均为 (8, 2048)，
    支持 float32 / float16，数据格式 ND，元素总数 16384。

二、开发环境与提交形态
    平台：CANNJudge（CANN 训练营-华南师范大学专场 20261114，Lesson 1）
    CANN 版本 9.0.0，编译架构 dav-2201。
    工程中仅 kernel.asc 可修改；main.asc 通过 #include 引入本文件，因此
    代码中不得出现 main()、#pragma once 或 include guard。
    核函数采用 __global__ __vector__ 直调形式，由主机侧 run_kernel 以
    &lt;&lt;&lt;blockDim, nullptr, stream&gt;&gt;&gt; 启动，blockDim 取 8。

三、实现思路
    1. 三级切分：总量 16384 先按核均分，每核 2048 个元素；核内再切成
       tileNum(8) × BUFFER_NUM(2) = 16 块，每块 128 个元素。
    2. 双缓冲：每个队列申请 2 块 UB 内存，使搬运与计算能够重叠，
       用以掩盖 DMA 延迟。
    3. 类型泛化：核函数写成模板类 KernelSub&lt;T&gt;，由主机侧按 dtype 实例化
       float 或 half，sizeof(T) 自动决定搬运量与 UB 申请量。
    4. 多核切分：每个核用 SetGlobalBuffer 将起始地址后移
       blockIdx × blockLength，使 8 个核各自处理互不重叠的数据段。

四、关键实现
    Init    ：计算 blockLength = totalLength / GetBlockNum()，
              tileLength = blockLength / tileNum / BUFFER_NUM，
              按核偏移设置 GlobalTensor，并为三个队列申请 UB 内存。
    CopyIn  ：DataCopy 将 x、y 各一块从 GM 搬入 UB，入队通知计算。
    Compute ：从队列取出 x、y，调用 AscendC::Sub 完成减法，结果入队，
              随后释放两个输入张量。
    CopyOut ：从输出队列取出结果，DataCopy 搬回 GM，并释放张量。
    注：DataCopy 目标参数在前，搬入为 (Local, Global, len)，
        搬出为 (Global, Local, len)。

五、正确性要点
    1. Sub(dst, src0, src1, len) 语义为 dst = src0 - src1，x 必须作为第一个
       源操作数（即 Sub 的第 2 个实参），y 作为第二个源操作数，顺序写反
       即为静默错误。
    2. AllocTensor 申请的 UB 必须成对 FreeTensor，否则循环中队列被占满。
    3. DataCopy 要求搬运字节数为 32 字节整数倍。本实现 fp32 单块
       128×4 = 512 字节，fp16 为 128×2 = 256 字节，核间偏移分别为
       8192 与 4096 字节，均满足对齐要求。
    4. 判题含 fp32、fp16 两个测试点，主机侧必须保留 dtype 分支；只实例化
       float 会导致 fp16 按 4 字节解析 2 字节数据而全线错位。
    5. 当前实现假定 totalLength 可被 blockDim × tileNum × BUFFER_NUM 整除，
       本题 16384 / 128 = 128 恰好整除；通用化需额外处理尾块。

六、验证结果
    提交 ID 1329368，状态 Pass，2/2 测试点通过，输出错误占比 0.00%，
    用时 6.15 us 与 6.09 us，当前最优记录为 1.74 us 与 1.65 us。
    减法为逐元素运算、无累加，结果与参考实现一致，无精度损失。

七、性能分析
    差距主要来自搬运粒度：每核 16 次循环约 48 条 DMA 指令，单次仅 512 字节，
    指令与同步开销压过了实际数据搬运时间。
    优化方向：将 tileNum 降为 1，单核一次搬完 2048 个元素（8 KB），
    循环由 16 次降为 1 次，DMA 指令由 48 条降为 3 条，预计延迟明显下降。

八、可扩展性
    三段式骨架可直接复用于其他 Vector 算子，仅需替换 Compute 中的一行：
    加法用 Add、乘法用 Mul、取大用 Max、单输入乘法用 Muls。
    如需支持 bf16 或 int32，在 run_kernel 中增加 dtype 分支即可，
    核函数本体无需改动。
