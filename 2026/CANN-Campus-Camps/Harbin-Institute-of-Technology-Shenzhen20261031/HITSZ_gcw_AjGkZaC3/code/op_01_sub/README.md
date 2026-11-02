# op_01_sub

提交团队: 哈尔滨工业大学（深圳）
提交者: gcw_AjGkZaC3


## 算子实现介绍

算子实现介绍：
本算子实现 Ascend C 矢量减法（z = x - y）。基于 8 核进行并行计算，Host 侧计算总长度并下发 Tiling 参数，Device 侧每个核处理 blockLength = totalLength / coreNum 的数据。
单核内采用双缓冲（BUFFER_NUM=2）和 tileNum 切分策略，通过 DataCopy 将数据搬运至 LocalTensor，调用 AscendC::Sub 接口完成减法计算，再搬回 GlobalTensor。支持 float32 和 float16 两种数据类型
