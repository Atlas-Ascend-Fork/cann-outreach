# expcustom

提交团队: 同济大学
提交者: keyiman944


## 算子实现介绍

基于 Ascend C 实现 Exp 算子，支持 float32、float16 和 ND 格式。针对题目输入形状（8, 2048），采用8核并行，每核处理2048个元素。使用静态 LocalTensor 分配输入、输出 UB，地址均按32字节对齐，省去 TPipe/TQue 初始化开销。搬入和搬出分别通过一次 DataCopy 调用完成，采用8个连续 burst 描述；计算使用原生 Exp 接口。通过 MTE2_V、V_MTE3 和 MTE3_S 事件保证搬运、计算及输出完成顺序。固定形状通过专用 tilingKey 选择编译期实现，避免运行时读取 tiling；其他规模保留通用分块与尾部处理路径。
