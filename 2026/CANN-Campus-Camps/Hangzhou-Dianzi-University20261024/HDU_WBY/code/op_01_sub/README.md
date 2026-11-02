# op_01_sub

提交团队: 杭州电子科技大学
提交者: Entropis


## 算子实现介绍

采用 Ascend C 直调模式实现 z = x - y。多核并行：按 GetBlockNum() 均分 totalLength，
各核用 GetBlockIdx() 定位起始偏移。核内 Tiling：blockLength 切为 tileNum 个 Tile，
配合 BUFFER_NUM=2 的 Double Buffer，loopCount = tileNum*BUFFER_NUM。
搬运计算：DataCopy 搬入 x、y 分块 → AscendC::Sub 逐元素相减 → DataCopy 写回结果。
通过 TQue 的 AllocTensor/EnQue/DeQue/FreeTensor 管理 UB 内存，实现搬算流水重叠。
模板化适配 fp16（half）与 fp32（float）两种数据类型。
