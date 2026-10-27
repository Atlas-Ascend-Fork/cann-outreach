# op_01_sub

提交团队: 上海理工大学
提交者: xiner11


## 算子实现介绍

算子功能
这是一个基于 Ascend C 原生直调的矢量减法（Vector Sub）算子。其核心功能是进行逐元素的减法运算：z = x - y。输入张量 x、y 与输出张量 z 的 shape 均为 (8, 2048)。

📊 支持的数据类型
通过 C++ 模板化（template &lt;typename T&gt;） 实现泛型编程。

在 Host 侧的 run_kernel 入口处，通过判断 TensorInfo 的 dtype，动态实例化具体的计算内核。

当前代码实现了对 float32（对应 float）和 float16（对应 half）两种数据类型的完整支持。

🧠 实现思路与算法机制
1. 多核切分与 Tiling 策略
算子采用了多核并行与核内切块相结合的 Tiling 策略，最大化利用 AI Core 资源：

总长度计算：Host 侧遍历输入的维度（numDims 和 shape），计算出总元素个数 totalLength（8 * 2048 = 16384）。

多核切分（Block 级）：固定启动 8 个 AI Core（blockDim = 8）。通过 totalLength / GetBlockNum() 算出每个核负责的 blockLength，并利用 GetBlockIdx() 计算各核在 Global Memory (GM) 中的起始偏移地址。

核内切块（Tile 级）：将单核负责的数据进一步切分为 tileNum = 8 块，以配合双缓冲机制进行细粒度流水线处理。每次搬运/处理的长度 tileLength = blockLength / tileNum / BUFFER_NUM。

2. 流水线设计 (Pipeline)
遵循 Ascend C 经典的三级流水线设计：

CopyIn：利用 DataCopy 将 x 和 y 从全局内存 (GM) 搬运到局部内存 (Local Memory / UB)。

Compute：从队列中取出数据，调用 Ascend C 内置的高阶 API AscendC::Sub 执行局部内存中的矢量减法。

CopyOut：将计算得到的 zLocal 从局部内存搬回全局内存 (GM)。

3. 性能优化点：双缓冲 (Double Buffer)
代码中定义了 BUFFER_NUM = 2，并在 InitBuffer 时为输入和输出队列分配了双倍大小的内存空间。

配合循环中的 loopCount = tileNum * BUFFER_NUM，当 AI Core 在执行第 i 块数据的 Compute 计算时，硬件 DMA 引擎可以同时在后台将第 i+1 块数据从 GM 搬运到 Local，从而实现数据搬运与计算的重叠（Overlap），有效隐藏内存访问延迟。

内存管理：严格遵循 AllocTensor -&gt; EnQue (入队) -&gt; DeQue (出队) -&gt; FreeTensor (释放) 的规范，确保流水线队列不阻塞、不泄漏。
