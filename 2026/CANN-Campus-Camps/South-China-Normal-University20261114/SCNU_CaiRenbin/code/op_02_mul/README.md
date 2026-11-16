# op_02_mul

提交团队: 华南师范大学
提交者: 蔡仁彬


## 算子实现介绍

## 3. 关键设计

### 3.1 Tiling 与多核数据切分

`MulTilingData`（`op_kernel/mul_tiling.h`）定义了两个关键参数：

```cpp
struct MulTilingData {
    uint32_t totalLength; // 输入张量总元素个数（8 * 2048 = 16384）
    uint32_t tileNum;     // 单核内分块数（本工程取 8）
};
```

Host 侧 `TilingFunc` 将输入 shape 各维相乘得到 `totalLength`，并配置启动核数为 8。
Kernel 侧按以下三层切分每个 AI Core 的职责范围：

```
blockLength = totalLength / GetBlockNum()        // 每核处理的数据长度
tileLength  = blockLength / tileNum / BUFFER_NUM // 每块（每次搬入/计算/搬出）的元素数
```

- `GetBlockNum()` 返回实际启动的核数（8 核并行）；
- 每个核通过 `GetBlockIdx()` 定位自己在全局内存中的起始偏移，实现互不重叠的并行切分；
- 单核内再将数据划分为 `tileNum × BUFFER_NUM` 个小块循环处理，块内使用统一局部缓冲区，
  数据量与 NPU 片上统一缓存（UB）容量解耦，可适配任意形状。

### 3.2 双缓冲流水

Kernel 侧定义了 `BUFFER_NUM = 2`，输入队列 `inQueueX`、`inQueueY` 与输出队列 `outQueueZ`
均申请 2 块缓冲（双缓冲）：

```cpp
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
```

配合 `AllocTensor / EnQue / DeQue / FreeTensor` 队列机制，实现“搬入上一块的同时计算当前块、
搬出结果”，使 Global Memory 数据搬移与矢量计算在时间上重叠，充分发挥流水并行能力。

### 3.3 模板化数据类型支持

`tiling_key_mul.h` 通过 `ASCENDC_TPL_ARGS_DECL` 声明 `DT_X` 可取 `C_DT_FLOAT16` 与
`C_DT_FLOAT` 两种模板参数，编译期自动生成两套实例；Host 侧通过
`ASCENDC_TPL_SEL_PARAM(context, DT_X)` 依据输入张量实际数据类型选中对应实例。

## 4. Kernel 侧实现详解（`op_kernel/mul.cpp`）

### 4.1 Init —— 初始化与缓冲区申请

根据 Tiling 参数完成地址偏移计算、全局内存句柄绑定和片上缓冲区初始化：

```cpp
this-&gt;blockLength = totalLength / AscendC::GetBlockNum();
this-&gt;tileNum = tileNum;
this-&gt;tileLength = this-&gt;blockLength / tileNum / BUFFER_NUM;
xGm.SetGlobalBuffer((__gm__ DT_X*)x + this-&gt;blockLength * AscendC::GetBlockIdx(), this-&gt;blockLength);
yGm.SetGlobalBuffer((__gm__ DT_X*)y + this-&gt;blockLength * AscendC::GetBlockIdx(), this-&gt;blockLength);
zGm.SetGlobalBuffer((__gm__ DT_X*)z + this-&gt;blockLength * AscendC::GetBlockIdx(), this-&gt;blockLength);
```

### 4.2 Process —— 主循环

循环次数为 `tileNum × BUFFER_NUM`，每一轮依次执行搬入、计算、搬出三段式处理：

```cpp
int32_t loopCount = this-&gt;tileNum * BUFFER_NUM;
for (int32_t i = 0; i &lt; loopCount; i++) {
    CopyIn(i);    // 搬入第 i 块数据
    Compute(i);   // 计算第 i 块数据
    CopyOut(i);   // 搬出第 i 块结果
}
```

### 4.3 CopyIn —— 数据搬入

从全局内存 `xGm`、`yGm` 的第 `progress * tileLength` 个元素起，各拷贝 `tileLength`
个元素到片上局部张量并入队：

```cpp
AscendC::LocalTensor&lt;DT_X&gt; xLocal = inQueueX.AllocTensor&lt;DT_X&gt;();
AscendC::LocalTensor&lt;DT_X&gt; yLocal = inQueueY.AllocTensor&lt;DT_X&gt;();
AscendC::DataCopy(xLocal, xGm[progress * this-&gt;tileLength], this-&gt;tileLength);
AscendC::DataCopy(yLocal, yGm[progress * this-&gt;tileLength], this-&gt;tileLength);
inQueueX.EnQue(xLocal);
inQueueY.EnQue(yLocal);
```

### 4.4 Compute —— 矢量乘法

从输入队列取出 `x`、`y` 块，调用矢量指令 `AscendC::Mul` 完成逐元素相乘，结果入输出队列，
随后释放输入张量：

```cpp
AscendC::LocalTensor&lt;DT_X&gt; xLocal = inQueueX.DeQue&lt;DT_X&gt;();
AscendC::LocalTensor&lt;DT_X&gt; yLocal = inQueueY.DeQue&lt;DT_X&gt;();
AscendC::LocalTensor&lt;DT_X&gt; zLocal = outQueueZ.AllocTensor&lt;DT_X&gt;();
AscendC::Mul(zLocal, xLocal, yLocal, this-&gt;tileLength);
outQueueZ.EnQue&lt;DT_X&gt;(zLocal);
inQueueX.FreeTensor(xLocal);
inQueueY.FreeTensor(yLocal);
```

### 4.5 CopyOut —— 结果搬出

将计算结果块写回全局内存 `zGm` 对应位置并释放输出张量：

```cpp
AscendC::LocalTensor&lt;DT_X&gt; zLocal = outQueueZ.DeQue&lt;DT_X&gt;();
AscendC::DataCopy(zGm[progress * this-&gt;tileLength], zLocal, this-&gt;tileLength);
outQueueZ.FreeTensor(zLocal);
```

### 4.6 核函数入口

读取 Tiling 数据并实例化算子对象启动执行：

```cpp
template &lt;typename DT_X&gt;
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul&lt;DT_X&gt; op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```

## 5. Host 侧实现详解（`op_host/mul.cpp`）

### 5.1 TilingFunc —— Tiling 计算

计算总元素数并写入 Tiling 数据，配置启动核数与 workspace（本算子无需额外 workspace）：

```cpp
const gert::StorageShape* x1_shape = context-&gt;GetInputShape(0);
int32_t data_sz = 1;
for (int i = 0; i &lt; x1_shape-&gt;GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape-&gt;GetStorageShape().GetDim(i);
tiling-&gt;totalLength = data_sz;
tiling-&gt;tileNum = TILE_NUM;        // 8
context-&gt;SetBlockDim(8);           // 启动 8 核
size_t *currentWorkspace = context-&gt;GetWorkspaceSizes(1);
currentWorkspace[0] = 0;
```

### 5.2 InferShape —— 形状推导

逐元素算子输出形状与输入完全一致：

```cpp
const gert::Shape* x1_shape = context-&gt;GetInputShape(0);
gert::Shape* z_shape = context-&gt;GetOutputShape(0);
*z_shape = *x1_shape;
```

### 5.3 InferDataType —— 数据类型推导

输出数据类型跟随输入数据类型：

```cpp
const auto inputDataType = context-&gt;GetInputDataType(0);
context-&gt;SetOutputDataType(0, inputDataType);
```

### 5.4 算子注册

在 `ops` 命名空间声明 `Mul : public OpDef`，配置输入/输出约束（`REQUIRED`，
`DT_FLOAT`、`DT_FLOAT16`，`FORMAT_ND`），挂接 Tiling 与 InferShape/InferDataType
回调，并通过 `AddConfig("ascend910b")` 注册到 AI Core 算子集。

## 6. 数据流与执行流程总结

```
Host 侧
  InferShape  → 输出 shape = 输入 shape
  InferDataType → 输出 dtype = 输入 dtype
  TilingFunc  → totalLength = 16384, tileNum = 8, blockDim = 8
                     │
                     ▼
Kernel 侧（8 核并行）
  每核 blockLength = 2048，分为 8 × 2 轮
  ┌───────────────────────────────────────────────┐
  │ CopyIn:  GM x,y ──DataCopy──▶ UB x,y (双缓冲) │
  │ Compute: UB x × UB y ──Mul──▶ UB z            │
  │ CopyOut: UB z ──DataCopy──▶ GM z              │
  └───────────────────────────────────────────────┘
```
