# op_03_tanh

提交团队: 华南师范大学
提交者: tematou


## 算子实现介绍

1. 算子功能
实现 PyTorch torch.tanh 功能，公式：
y
=
e
x
−
e
−
x
e
x
+
e
−
x
y= 
e 
x
 +e 
−x
 
e 
x
 −e 
−x
 

 。
输入/输出形状均为 (8, 2048)，支持 float16 / float32，格式 ND。

2. Host 侧 (tanh_tiling.cpp)

Tiling 切分：根据输入总元素（16384）和 UB 容量，计算单核搬运数据块大小。

临时空间：调用 GetTanhMaxMinTmpSize 获取计算所需临时空间（Tmp Buffer），并写入 TilingData。

3. Kernel 侧 (tanh.h)

数据搬运：将数据从全局内存（GM）分块搬运至统一缓冲区（UB）。

核心计算：申请 Tmp Buffer，调用 AscendC::Tanh 高阶接口完成计算（支持模板泛化以兼容 fp16/fp32）。

结果回写：计算完成后将数据搬出至输出张量。

4. 验证流程
网页点击“提交代码” 
→
→ 平台编译并跑验证集 
→
→ 精度一致显示“PASS” 
→
→ 填写学校及 Gitcode 信息提交开源仓。
