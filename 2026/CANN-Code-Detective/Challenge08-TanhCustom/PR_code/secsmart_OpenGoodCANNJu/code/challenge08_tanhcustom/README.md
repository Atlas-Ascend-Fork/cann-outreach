# challenge08_tanhcustom

提交团队: 闪捷信息科技有限公司
提交者: gcw_LxlU8H9L


## 算子实现介绍

基于Ascend C高精度Tanh指令实现Tanh激活算子（y=tanh(x)）。op_host侧完成形状推导与Tiling函数：按输入dtype选择模板实例（float16/float32），总元素数按32核均匀切分（blockFactor=totalNum/32），核内再按ubFactor分片搬入UB；op_kernel侧采用CopyIn→Compute→CopyOut三段式流水线，BUFFER_NUM=2双缓冲使搬运与计算流水并行，核内调用高精度Tanh指令逐元素计算并写回GM。已在昇腾910B实机完成编译与精度验证（tanh(1)=0.761594与理论值一致）
