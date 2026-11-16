# op_03_tanh

提交团队: 华南师范大学
提交者: gcw_vuifW5gj


## 算子实现介绍

定义与推导：tanh_def.cpp 注册单输入 x、单输出 y（float32/float16，ND）；tanh_infershape.cpp 输出 shape 直接拷贝输入 shape。
Tiling 数据与 Key：tanh_tiling_data.h 含 totalLength/tileNum/tileLength 三字段；tanh_tiling_key.h 按 dtype 声明 DT_X 模板参数分派 float/half 实例。
Tiling 实现：tanh_tiling.cpp 逐维累乘得总元素数，按 MIN_PER_CORE=2048 与平台核数取 min 定 blockDim，写 tileLength=4096，workspace 置 0。
核函数入口：tanh.cpp 取 tiling 后构造 NsTanh::Tanh&lt;DT_X&gt; 并调 Init/Process。
多核与 UB 切分：Init 按核数均分并对齐 32B，末核吸收余量；tileLength 取 tiling 值与 curLength 较小者对齐，各队列与 5 个 TBuf 按 allocLength 分配。
流水与尾块：Process 预取 2 个 tile 后循环“计算→预取→写出”；GetCopyLen 按剩余长度搬算，末块不足时只处理有效长度。
数值保护：Compute 先 Maxs/Mins clamp 到 [-10,10]，防 float16 下 exp 溢出。
小算子拼接：exp(x)、exp(-x) 分别存 tmp2/tmp3，分母 tmp4=tmp2+tmp3、分子 tmp1=tmp2-tmp3，y=tmp1/tmp4，全程无 tanh API。
