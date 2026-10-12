# challenge06_relucustom

提交团队: 江苏润和软件股份有限公司
提交者: longvoyage


## 算子实现介绍

单核实现，SetBlockDim(1)，不进行多核切分；无 system workspace（WS_SYS_SIZE = 0）。
UB 切分：按 UB block size（32B）对齐，借助 CeilAlign/FloorAlign + GetUbBlockSize 将数据量向上/向下对齐到 32B 整数倍；MIN_SPLIT_THRESHOLD = 1024 作为可切分的最小阈值，TYPE_SIZE = 4 用于按元素字节数换算 UB 可容纳的元素数。
TilingKey 分派：按输入 dtype 选择调度模式——FP16/BF16 走 RELU_TPL_SCH_MODE_0（半精度模板），其余（如 FP32）走 RELU_TPL_SCH_MODE_1，通过 GET_TPL_TILING_KEY 生成模板 tiling key。
