# op_01_sub

提交团队: 华南师范大学
提交者: 王文挺


## 算子实现介绍

# Sub 算子实现介绍

## 1. 算子功能
Sub 算子实现逐元素减法：z[i] = x[i] - y[i]。

## 2. 输入输出
- 输入 x、y：与输出 z 同 shape、同 dtype。
- 输出 z：逐元素减法结果。
- 当前支持 dtype：fp16 使用 half 实例；fp32 及其他类型默认使用 float 实例。

## 3. Tiling 与切分策略
totalLength 由输入 shape 乘积得到。
多核切分：
  blockLength = totalLength / GetBlockNum()
  每个核处理连续 blockLength 个元素，GM 起始偏移为 blockLength * GetBlockIdx()。
单核分块：
  tileLength = blockLength / tileNum / BUFFER_NUM
  loopCount = tileNum * BUFFER_NUM
当前 tileNum = 8，BUFFER_NUM = 2，QUEUE_DEPTH = 1。

## 4. Kernel 执行流程
KernelSub&lt;T&gt; 包含 Init、Process、CopyIn、Compute、CopyOut。
Process 循环 loopCount 次：
  CopyIn：将 x、y 当前块搬入 VECIN 队列；
  Compute：调用 AscendC::Sub 计算 zLocal = xLocal - yLocal；
  CopyOut：将 zLocal 搬出到 GM。
队列使用 AllocTensor、EnQue、DeQue、FreeTensor 管理 UB 内存。

## 5. Host 启动
Host 侧根据 shape 计算 totalLength，设置 tileNum = 8、blockDim = 8。
根据 dtype 选择 sub_custom&lt;half&gt; 或 sub_custom&lt;float&gt;，通过 stream 启动。

## 6. 约束与边界
当前实现要求 totalLength 可被 blockDim 整除，且 blockLength 可被 tileNum * BUFFER_NUM 整除。
当前未处理尾块；非 fp16 类型默认走 float 实例；blockDim 固定为 8，未使用 availableCoreNum 动态调整。

## 7. 优化方向
增加尾块处理；按 availableCoreNum 设置 blockDim；补充 bf16/int 等 dtype；调整 tileNum/BUFFER_NUM/QUEUE_DEPTH 提升流水；检查 DataCopy 对齐。
