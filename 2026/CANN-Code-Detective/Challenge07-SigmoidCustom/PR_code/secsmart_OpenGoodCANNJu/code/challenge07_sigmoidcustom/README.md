# challenge07_sigmoidcustom

提交团队: 闪捷信息科技有限公司
提交者: gcw_LxlU8H9L


## 算子实现介绍

基于Ascend C实现Sigmoid激活算子（y=1/(1+exp(-x))）。op_host侧完成形状推导与Tiling函数：按输入dtype选择模板实例（float16/float32），总元素数按8核均匀切分（blockFactor=totalNum/8），核内再按ubFactor分片搬入UB；op_kernel侧采用CopyIn→Compute→CopyOut三段式流水线，内部统一转float并使用0.5+0.5*tanh(x/2)数值稳定公式（经Muls/Tanh/Adds指令组合实现，避免fp16大负数exp溢出），结果写回GM后Cast回原dtype
