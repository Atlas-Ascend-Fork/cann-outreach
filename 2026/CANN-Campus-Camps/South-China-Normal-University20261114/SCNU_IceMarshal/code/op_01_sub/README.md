# op_01_sub

提交团队: 华南师范大学
提交者: 2401_87109648


## 算子实现介绍

矢量减法算子实现介绍
实现方式
采用单核实现，启动配置为 blockDim = 1。单核内将 16384 个元素划分为 4 个数据块，每块 4096 个元素，按块循环执行搬运、计算、回写。核函数由 KernelSub&lt;T&gt; 模板实例化，float 与 half 共用同一套流程。

UB 切分与对齐
UB 按 32 B 对齐切分，具体划分如下：

输入缓冲由 x、y 共用，容量为 2 × tileLength × sizeof(T)；x 占起始段，y 占偏移 tileLength 处。
输出缓冲容量为 tileLength × sizeof(T)。
tileLength = 4096，fp32 每块 16384 B、fp16 每块 8192 B，y 的偏移量与各缓冲容量均为 32 B 的整数倍。
fp32 下输入缓冲 32 KB、输出缓冲 16 KB，共 48 KB；fp16 下分别为 16 KB、8 KB。
BUFFER_NUM = 1、QUEUE_DEPTH = 1，单缓冲配合逐块串行处理。
Tiling 策略
宿主侧先由 info_x 各维求积得到 totalLength = 16384，再按 BLOCK_SIZE = 32 对齐：

计算总字节数，fp32 为 65536 B，fp16 为 32768 B。
上取整到 32 的整数倍得到 alignedBytes，再换算为元素数 alignedLength。
以每块 4096 个元素计算 tileNum = alignedLength / 4096，得到 4 块。
核内 tileLength = blockLength / tileNum / BUFFER_NUM，单核下 blockLength 等于 alignedLength，结果为 4096。
本形状的对齐运算不产生填充元素，alignedLength 与实际张量长度一致。

数据通路与同步
每个数据块依次执行三个阶段：

搬入：两次 DataCopy 将 x、y 分别写入输入缓冲的首段与偏移段，随后统一执行一次 EnQue。
计算：DeQue 取得输入缓冲，调用 Sub 完成 4096 个元素的减法，结果写入输出缓冲并入队。
回写：从输出队列取出结果，DataCopy 写回全局内存。
每块产生一组 MTE2→V 同步与一组 V→MTE3 同步。x、y 合并到同一队列，使搬入侧只需一次入队事件，而非两次。

数据类型处理
宿主侧根据 TensorInfo.dtype 选择模板实例，取值为 1 时实例化 half，其余情况实例化 float；同时以 2 或 4 作为元素字节宽度参与对齐计算。设备侧的计算与搬运接口由模板参数推导数据类型。
