/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

template <typename DT_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    // 小 shape 单 task 路径不构造 TPipe/TQue，规避固定缓冲框架开销。
    if (!NsRelu::NsReluDirect::TryRunDirect<DT_X>(x, y, tilingData)) {
        NsRelu::Relu<DT_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
#else
    NsRelu::Relu<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
#endif
}
