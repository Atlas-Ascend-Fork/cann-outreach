# challenge06_relucustom

提交团队: 闪捷信息科技有限公司
提交者: gcw_LxlU8H9L


## 算子实现介绍

基于Ascend C实现ReLU激活算子（y=max(x,0)）。op_host侧完成形状推导与Tiling函数：按输入dtype选择模板实例（float16/float32），总元素数按8核均匀切分（blockFactor=totalNum/8），核内再按ubFactor分片搬入UB；op_kernel侧采用CopyIn→Compute→CopyOut三段式流水线，DataCopy搬入UB后调用Relu指令逐元素计算并写回GM。已通过平台全部测试点验证。
