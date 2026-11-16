# op_01_sub

提交团队: 华南师范大学
提交者: 王佳豪


## 算子实现介绍

基于Ascend C实现(8,2048) ND格式逐元素向量减法z=x-y，代码提供float32/float16模板实例。8核并行，每核2048元素；tileNum=1、BUFFER_NUM=1，整块DataCopy搬入，经AscendC::Sub计算后搬回GM。平台2个测试点全部Pass，输出错误占比均为0%，耗时分别为3.06μs、2.98μs。
