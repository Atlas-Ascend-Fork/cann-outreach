# challenge08_tanhcustom

提交团队: 江苏润和软件股份有限公司
提交者: longvoyage


## 算子实现介绍

执行并行：单核实现，SetBlockDim(1)，不做多核切分；无 system workspace（WS_SYS_SIZE = 0）。
UB 切分：按 UB block size（32B）对齐，通过 CeilAlign/FloorAlign + GetUbBlockSize 将数据量对齐到 32B 整数倍；MIN_SPLIT_THRESHOLD = 1024 为最小可切分阈值，TYPE_SIZE = 4 按元素字节数换算 UB 可容纳元素数。
TilingKey 分派：按输入 dtype 选择模板调度模式——FP16/BF16 → TANH_TPL_SCH_MODE_0，其余（FP32 等）→ TANH_TPL_SCH_MODE_1，经 GET_TPL_TILING_KEY 生成 tiling key。
