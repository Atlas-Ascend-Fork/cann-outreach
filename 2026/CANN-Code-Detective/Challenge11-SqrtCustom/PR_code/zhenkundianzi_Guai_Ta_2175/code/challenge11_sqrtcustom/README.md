# challenge11_sqrtcustom

提交团队: 武汉贞坤电子有限公司
提交者: TangYunCheng


## 算子实现介绍

本算子实现 PyTorch 原生 torch.sqrt（y=√x，约定输入 x≥0），支持 (8,2048) 的 float32/float16 ND 张量，输出同形状同类型。
Kernel 侧采用 Ascend C 模板类 Sqrt&lt;T&gt;：
Init 绑定输入/输出 GlobalTensor 并按 ubFactor 初始化双缓冲 TQue；
Process 先按 blockFactor 将总元素切分到各 AI Core，再在核内按 ubFactor 分块循环执行 CopyIn、Compute、CopyOut，其中 Compute 调用 AscendC::Sqrt 完成逐元素开方。
Host 侧注册 SqrtTilingData，依据平台 coreNum、ubSize 和输入 dtype 计算 totalNum、blockFactor、ubFactor，做 32B 对齐并设置 BlockDim；
按 dtype 选择 tilingKey（float16/bf16 走 mode0，float32 走 mode1），workspace 为 0；
Infershape 将输出形状直接设为与输入一致。
