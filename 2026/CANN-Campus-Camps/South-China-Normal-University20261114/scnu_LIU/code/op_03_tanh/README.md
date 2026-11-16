# op_03_tanh

提交团队: 华南师范大学
提交者: l_l1114812502


## 算子实现介绍

Tiling策略：由于输入固定为(8, 2048)，共16384个元素，Host侧配置SetBlockDim为8，实现单核处理单行2048个元素的完美负载均衡（blockFactor与ubFactor均为2048），无需处理复杂的尾块逻辑。
Kernel实现：

    数值优化：采用 tanh(x) = 2 / (1 + exp(-2x)) - 1 的变形公式，相比原生公式减少了一次Exp计算，提升性能的同时兼顾了数值稳定性。

    精度保障：输入若为FP16/BF16，在进入UB后统一Cast升至FP32进行运算（Muls、Exp、Adds、Div），最后再Cast回原类型，规避了半精度计算溢出问题，确保结果与PyTorch原生完全对齐。

    内存管理：按32B对齐原则在UB上硬切分了x、a、b、y四个独立无重叠的LocalTensor空间，配合PipeBarrier保障流水线同步。
    TilingKey：通过Host侧识别Data Type，动态下发 schMode 模板参数（0对应FP16/BF16，1对应FP32），实现不同数据类型分支的精准派发。
