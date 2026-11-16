# op_04_relu

提交团队: 华南师范大学
提交者: gcw_vuifW5gj


## 算子实现介绍

定义与推导：relu_def.cpp 注册单输入 x、单输出 y（float32/float16，ND）；relu_infershape.cpp 输出 shape 直接拷贝输入 shape。
Tiling 数据：relu_tiling_data.h 三字段 totalLength（总元素数）/tileNum/tileLength（单次 UB 处理长度）。
Tiling Key：relu_tiling_key.h 按输入 dtype 声明 DT_X 模板参数，编译期分派 float / half 实例。
Tiling 实现：relu_tiling.cpp 逐维累乘得总元素数，按 MIN_PER_CORE=2048 与平台核数取 min 定 blockDim，写 tileLength=4096，workspace 置 0。
多核切分：Init 按核数均分并对齐 32B，末核吸收余量，每核设 GM 偏移与 curLength。
UB 分块：tileLength 取 tiling 值与 curLength 较小者并对齐，allocLength 为对齐后分配长度，输入/输出各一个双缓冲 TQue。
双缓冲流水：Process 先预取 2 个 tile，再循环“计算当前 → 预取下一 → 写出当前”，实现 CopyIn/Compute/CopyOut 流水。
尾块处理：GetCopyLen 按 progress*tileLength 算剩余长度，末块不足时只搬/算有效长度，避免越界。
核心计算：Compute 用一行 AscendC::Maxs(yLocal, xLocal, 0, copyLen) 实现 y = max(0, x)，无临时缓存、无溢出保护需求。
类型支持：relu.cpp 模板入口实例化 float 与 half，配合 relu_def.cpp 的 dtype 列表与 TilingKey 共同保证 float16 输入输出。
