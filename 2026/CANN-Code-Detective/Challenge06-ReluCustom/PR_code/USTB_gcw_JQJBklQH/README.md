# 北京科技大学

## 团队信息

- 提交者: 北京CANN生态技术创新组肖辉
- 身份: 学生
- 单位: 北京科技大学

## 成员

- 北京CANN生态技术创新组肖辉 (gcw_JQJBklQH): 提交者

## 算子: challenge06_relucustom

该算子实现了 PyTorch torch.nn.functional.relu（y = max(0, x)）的昇腾 NPU 版本，采用标准开源仓三层结构：主机侧 op_host 负责算子注册、形状推导（输出 shape 复制输入）以及 ReluTilingFunc tiling 计算（按输入 dtype 选 tilingKey、逐维相乘得 totalLength、设 tileNum=16、启动 8 核、workspace=0）；设备侧 op_kernel 用模板参数 DT_X 实现一套代码编译出 float/float16 两个实例，在 Relu&lt;DT_X&gt; 类里按 Vector 编程范式的 CopyIn → Compute → CopyOut 三段式流水执行——多核按 blockLength + GetBlockIdx() 切分各自的数据段，经 DataCopy 把数据从 GM 搬入 UB，用硬件 AscendC::Relu 指令逐元素计算，再搬回 GM；全程严格遵循 GM→UB→寄存器的内存层级，并通过 TQue 队列管理缓冲生命周期（BUFFER_NUM 留给双缓冲优化）。构建通过、op_host/op_kernel UT 全过、精度比对 MERE=0/MARE=0 与 PyTorch golden 完全一致。
