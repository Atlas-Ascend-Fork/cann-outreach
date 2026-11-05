# op_01_sub

提交团队: 上海理工大学
提交者: jibenji


## 算子实现介绍

在 kernel.asc 内补充以下实现：

struct SubCustomTilingData：tiling 参数结构体
class KernelSub：核函数主体类实现
__global__ __vector__ void sub_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, SubCustomTilingData tiling)：核函数入口
extern "C" void run_kernel(GM_ADDR x, const TensorGroupInfo&amp; info_x, GM_ADDR y, const TensorGroupInfo&amp; info_y, GM_ADDR z, const TensorGroupInfo&amp; info_z, int64_t availableCoreNum, aclrtStream stream)：host 侧核函数启动
