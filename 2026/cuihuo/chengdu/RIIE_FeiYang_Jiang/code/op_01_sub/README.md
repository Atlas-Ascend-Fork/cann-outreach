# op_01_sub

提交团队: 成都智慧企业发展研究院
提交者: 蒋飞扬


## 算子实现介绍

整体形态：Ascend C 直调（&lt;&lt;&lt;&gt;&gt;&gt;）单文件算子。kernel.asc 被 main.asc 直接
  #include，因此一个翻译单元里同时包含 host 端启动逻辑和 device 端
  kernel；入口是框架固定的 run_kernel（kernel.asc:92）。
  
  1. Host 端：run_kernel（kernel.asc:92-125）

  ┌──────────┬──────────────────────────────────────────────────────────────────┐
  │   步骤   │                               做法                               │
  ├──────────┼──────────────────────────────────────────────────────────────────┤
  │ 元素总数 │ numelOf 遍历 TensorInfo.shape/numDims 求乘积，优先用             │
  │          │ info_z，回退 info_x（kernel.asc:79-90）                          │
  ├──────────┼──────────────────────────────────────────────────────────────────┤
  │ dtype    │ info_x.tensors[0].dtype，1 → half，其余 →                        │
  │ 派发     │ float，模板实例化两个版本（kernel.asc:117-124）                  │
  ├──────────┼──────────────────────────────────────────────────────────────────┤
  │ 多核切分 │ numBlocks = min(availableCoreNum, ceil(total/1024))，每核        │
  │          │ blockLength = ceil(total/numBlocks)（kernel.asc:110-115）        │
  ├──────────┼──────────────────────────────────────────────────────────────────┤
  │ 启动     │ sub_custom&lt;T&gt;&lt;&lt;&lt;numBlocks, nullptr, stream&gt;&gt;&gt;(...)，blocks 个    │
  │          │ AIV 并行                                                         │
  └──────────┴──────────────────────────────────────────────────────────────────┘

  样例下 16384 元素 → 16 个 block × 1024 元素/核。

  2. Device 端：sub_custom&lt;T&gt;（kernel.asc:15-77）

  纯向量核（__global__ __aicore__ __vector__），结构化流水：

  - 资源：TPipe + 2 个 VECIN 队列（x、y）+ 1 个 VECOUT 队列（z），深度 2，每块
  8KB，共 6×8KB=48KB UB（kernel.asc:12-30）。
  - 任务划分：blockStart = blockLength * GetBlockIdx()，本核负责 [blockStart, 
  min(blockStart+blockLength, total))，空核直接返回（kernel.asc:32-39）。
  - 分块循环（kernel.asc:51-76）：
    a. DataCopyPad 把 x、y 的当前 tile 搬入 UB（MTE2）；
    b. AscendC::Sub(zLocal, xLocal, yLocal, elementNum) 向量相减（V）；
    c. DataCopyPad 把 z 搬回 GM（MTE3）。
  - 同步：全部交给队列语义——EnQue/DeQue 建立 MTE2→V→MTE3 依赖，FreeTensor
  释放缓冲，无手写 SetFlag/WaitFlag。
  - 尾块：每轮动态计算 elementNum，DataCopyPad 按实际字节数搬运，非 32B
  对齐也能正确读写。
  - dtype 无关：T 决定 sizeof(T) 与 tile 元素数，fp16/fp32 共用同一份代码。

  3. 为什么这样写

  - 16K 元素很小，MIN_ELEMS_PER_CORE=1024 避免过度拆核导致启动开销占比过高；tile 8KB
   保证 UB 占用远低于上限且天然对齐。
  - 逐元素线性映射 z[i]=x[i]-y[i]，与 golden x-y（fp64 计算后回舍入）在 fp32/fp16
  下结果精确一致，实测 mismatched=0/16384。
  - 首版曾在 device 代码里用 std::min
  编译失败，已改为三元比较（kernel.asc:52-55）——device 侧不能依赖 host 标准库函数。
